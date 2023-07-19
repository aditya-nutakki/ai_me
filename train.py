import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from models import GPTConfig, GPT
from utils import *
import config as c

dv = WADataValidator()

device = c.device
block_size = c.block_size
weight_decay = c.weight_decay
learning_rate = c.learning_rate
beta1, beta2 = c.beta1, c.beta2
eval_iters = c.eval_iters
warmup_iters = c.warmup_iters
lr_decay_iters, min_lr, decay_lr = c.lr_decay_iters, c.min_lr, c.decay_lr
iter_num, eval_interval = c.iter_num, c.eval_interval
master_process = c.master_process
out_dir = c.out_dir
eval_only, always_save_checkpoint = c.eval_only, c.always_save_checkpoint
gradient_accumulation_steps = c.gradient_accumulation_steps
grad_clip = c.grad_clip
max_iters = c.max_iters
best_val_loss = 1e8

os.makedirs(out_dir, exist_ok=True)
print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

gptconf = GPTConfig()
model = GPT(gptconf).to(device)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)

model.to(device)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # X, Y = dv.get_batch(mode="val")
            X, Y = get_tensor_batch(mode="val")
            X, Y = X.to(device), Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_tensor_batch(mode = "train"):
    x, y = dv.get_batch(mode=mode)
    return torch.stack(x).to(device), torch.stack(y).to(device)


# X, Y = dv.get_batch() # fetch the very first batch from training set
X, Y = get_tensor_batch(mode="train")
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break


    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            # X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

        X, Y = get_tensor_batch(mode="train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
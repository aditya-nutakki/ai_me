import torch
import torch.nn.functional as F
from models import GPT, GPTConfig
import config as c
from utils import stoi, itos

block_size = c.block_size

def generate(start_token = " ", max_tokens=111):
    
    with torch.no_grad():
        token = torch.tensor(stoi(start_token), dtype=torch.long).view(1,-1).to(device)
        full_token = torch.tensor(stoi(start_token), dtype=torch.long).view(1,-1).to(device)
        # print(token.shape)
        # print(token)
        for l in range(len(start_token), max_tokens-1):
            # print(token.shape, l)
            if l > block_size:
                token = token[:, abs(l-block_size):]
            ops, _ = model(token)
            ops = ops[:, -1, :]
            ops = F.softmax(ops, dim = -1)
            most_likely_char = torch.multinomial(ops, num_samples = 1)
            token = torch.cat((token, most_likely_char), dim = 1)
            full_token = torch.cat((token, most_likely_char), dim = 1)
        
        # print(token)
    return itos(full_token.cpu().detach().numpy().flatten().tolist())

def encode_prompt(prompt):
    return torch.tensor(stoi(prompt), dtype=torch.long).view(1, -1).to(device)

def decode_tensor(tensor):
    return itos(tensor.cpu().detach().numpy().flatten().tolist())

device = c.device
model_path = "./out/ckpt_2000_24_24_264.pt"

model = GPT(GPTConfig())

ckpt = torch.load(model_path)
# print(ckpt.keys())
if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict)
model = model.to(device)
print("Loaded model ... ")
print()
input_prompt = " "
gen = model.generate(idx = encode_prompt(input_prompt), max_new_tokens = 500)
# print(gen, gen.shape)
gen = decode_tensor(gen)
print(gen)


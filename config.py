block_size = 96
batch_size = 8
text_file_name = "./datasets/combined.txt" # file to train model from
vocab_path = "./datasets/vocab.txt"
train_split = 0.9


# hyperparams
device = "cuda" # can be cpu as well
out_dir = './out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
iter_num = 0
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
master_process = True

gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
# model
n_layer = 18
n_head = 18
n_embd = 216
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

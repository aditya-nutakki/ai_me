import torch
import os
import config as c
from utils import WADataValidator


# basically a utils file but including pytorch.

dv = WADataValidator(mode="train")
train_data = dv.train_data
val_data = dv.val_data

print(len(train_data), len(val_data))
print(dv.encode(train_data[:10]))
print(dv.decode(dv.encode(train_data[:10])))

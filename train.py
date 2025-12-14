import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from transformers import SamModel, SamProcessor
from transformers import CLIPVisionModel
import torchvision.transforms as T
import torch.nn.functional as F

import math
import os
import tiktoken

from torch.utils.data import DataLoader, Dataset

from deepencoder import CLIP_modified, conv_block, DeepEncoder
from dataloader import OCR_dataset, ocr_collate
from tqdm import tqdm

from helper import text_to_token_ids, token_ids_to_text, download_and_load_gpt2
from knowledge_transfer import load_weights_into_gpt_modified
from model import GPTModel

from pipeline import *

from build import *

import wandb

import warnings
warnings.filterwarnings("ignore")

########################
########################

train_frac = 0.8
test_frac  = 0.15
batch_size = 1
device = "cpu"

num_epochs = 2

lr = 0.00005
weight_decay = 0.1

load_pretained = False

wandb_logging = True

if wandb_logging:
    run = wandb.init(
        entity="mafaz03",
        project="OCR_deepseek",

        name="logging",
        
        config={
            "learning_rate": lr,
            "weight_decay" : weight_decay,
            "train_frac"   : train_frac,
            "test_frac"    : test_frac,
            "batch_size"   : batch_size,
            "num_epochs"   : num_epochs,
            "device"       : device,
        },
    )
    print("wandb initialized")
else: run = None

########################
########################

print("Initialising tokenizer (modified)")
tokenizer = tiktoken.get_encoding('gpt2')

special_tokens = {"<image>": tokenizer.n_vocab+1}
tokenizer_modified = tiktoken.Encoding(
    name="gpt2_with_image",
    pat_str=tokenizer._pat_str,
    mergeable_ranks=tokenizer._mergeable_ranks,
    special_tokens={**tokenizer._special_tokens, **special_tokens}
)
vocab_size = tokenizer_modified.n_vocab
print("Tokenizer Loaded\n")

print("Loading Dataset")
files = os.listdir('dataset')
l = len(files)

# train_frac = 0.8
# test_frac  = 0.15

train_pos = int(l * train_frac)
test_pos  = int(l * test_frac)

train_files = files[: train_pos]
test_files = files[train_pos : train_pos + test_pos]
val_files  = files[train_pos + test_pos : ]

print("Dataset sizes")
len(train_files), len(test_files), len(val_files)

train_dl = DataLoader(
           dataset=OCR_dataset(
               dataset_file_name = 'dataset',
               files = train_files,
               tokenizer = tokenizer_modified
               ),
           batch_size=batch_size,
           shuffle=True,
           collate_fn=ocr_collate,
           pin_memory=True,
           drop_last = True
       )

test_dl  = DataLoader(
           dataset=OCR_dataset(
               dataset_file_name = 'dataset',
               files = test_files,
               tokenizer = tokenizer_modified
               ),
           batch_size=batch_size,
           shuffle=False,
           collate_fn=ocr_collate,
           pin_memory=True,
           drop_last = True
       )

val_dl  =  DataLoader(
           dataset=OCR_dataset(
               dataset_file_name = 'dataset',
               files = val_files,
               tokenizer = tokenizer_modified
               ),
           batch_size=batch_size,
           shuffle=False,
           collate_fn=ocr_collate,
           pin_memory=True,
           drop_last = True
       )

print("Dataloader shapes")
one_batch  = next(iter(train_dl))

one_batch_input_ids  = one_batch["input_ids"]
one_batch_target_ids =  one_batch["target_ids"]
one_batch_images     = one_batch["images"]

print(f"input_ids: {one_batch_input_ids.shape}")
print(f"target_ids: {one_batch_target_ids.shape}")
print(f"images: {one_batch_images.shape}")
print("Dataloader Loaded\n")

print("Loading encoder")
deep_encoder = build_encoder(device)
print("Encoder loaded\n")

print("Loading decoder")
NEW_CONFIG, gpt2 = build_decoder(tokenizer, tokenizer_modified)
print("Decoder loaded\n")

optimizer = torch.optim.AdamW(gpt2.parameters(), lr=lr, weight_decay=weight_decay)

if __name__ == "__main__":
    print("### TRAINING STARTED ###")
    train_losses, val_losses, tokens_seen = train(train_loader   = train_dl,
                                                  val_loader     = val_dl,
                                                  deep_encoder   = deep_encoder,
                                                  deep_decoder   = gpt2,
                                                  cfg            = NEW_CONFIG,
                                                  device         = device,
                                                  num_epochs     = num_epochs,
                                                  eval_freq      = 5,
                                                  eval_itter     = 2,
                                                  tokenizer      = tokenizer_modified,
                                                  optimizer      = optimizer,
                                                  max_new_tokens = 10,
                                                  save_itter     = 1,
                                                  save_path      = "gpt2/OCR_finetuned/gpt2_774M_finetuned.pth",
                                                  load_pretained = load_pretained,
                                                  verbose        = True,
                                                  wandb_logging  = wandb_logging,
                                                  run            = run)
    
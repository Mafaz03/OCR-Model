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

def build_encoder(device):
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    deep_encoder = DeepEncoder(sam_model = sam_model, clip_model = clip_model)

    return deep_encoder

def build_decoder(tokenizer, tokenizer_modified):
    GPT_CONFIG_124M = {
    "vocab_size"     : tokenizer.n_vocab,     # 50257
    "context_length" : 1024,                  # The maximum number of tokens the model can process at once
    "embedding_dim"  : 768,                   # The number of features used to represent each token 
    "n_heads"        : 12,
    "n_layers"       : 12,                    # How many transformer blocks
    "drop_rate"      : 0.1,
    "qkv_bias"       : False
    }

    model_configs = {
        "gpt2-small (124M)": {"embedding_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"embedding_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"embedding_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"embedding_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_name = "gpt2-large (774M)"

    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, 
                    "qkv_bias": True, 
                    "vocab_size": tokenizer_modified.n_vocab,
                    "vision_dim": 1280})

    settings, params = download_and_load_gpt2(model_size="774M", models_dir="gpt2")
    gpt2 = GPTModel(NEW_CONFIG)
    load_weights_into_gpt_modified(gpt2, params)
    return NEW_CONFIG, gpt2

def train(train_loader, val_loader, 
          deep_encoder, deep_decoder, cfg,
          optimizer, device, num_epochs, 
          eval_freq, eval_itter, 
          tokenizer, 
          verbose = True, max_new_tokens = 50, 
          save_itter = 5, 
          save_path = "gpt2/OCR_finetuned/gpt2_774M_finetuned.pth",
          load_pretained = True,
          ):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    if load_pretained:
        checkpoint = torch.load(save_path, map_location="cpu")

        epoch_continue = checkpoint["epoch"]
        deep_decoder.load_state_dict(checkpoint["model_state"])
        print("loaded")
        
    else:
        epoch_continue = 0

    for epoch in range(epoch_continue, num_epochs + epoch_continue):
        deep_decoder.train()
        local_step = 0
        for idx, a in enumerate(train_loader):
            
            input_batch  = a["input_ids"]
            target_batch = a["target_ids"]
            image_batch  = a["images"]

            optimizer.zero_grad()

            loss = calc_loss_batch(pipline      = vision_pipeline,
                                   deep_encoder = deep_encoder,
                                   deep_decoder = deep_decoder,
                                   input_batch  = input_batch,
                                   target_batch = target_batch,
                                   image_batch  = image_batch,
                                   tokenizer    = tokenizer
                                   )
            
            loss.backward()

            optimizer.step()
            tokens_seen = input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                deep_decoder.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(dataloader   = train_loader,
                                                  deep_encoder = deep_encoder,
                                                  deep_decoder = deep_decoder,
                                                  tokenizer    = tokenizer,
                                                  num_batches  = eval_itter,
                                                  pipeline     = vision_pipeline)
                
                    val_loss   = calc_loss_loader(dataloader   = val_loader,
                                                  deep_encoder = deep_encoder,
                                                  deep_decoder = deep_decoder,
                                                  tokenizer    = tokenizer,
                                                  num_batches  = eval_itter,
                                                  pipeline     = vision_pipeline)
                deep_decoder.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )
            local_step += 1
            if local_step % save_itter == 0:
                checkpoint = {
                                "epoch"          : epoch,
                                "model_state"    : deep_decoder.state_dict(),
                             }

                torch.save(checkpoint, save_path)
                if verbose: print("saved")
                
        # print some samples
        if verbose:
            text = generate_text(
                        deep_encoder   = deep_encoder,
                        gpt2           = deep_decoder,
                        projector      = deep_encoder.projector,
                        tokenizer      = tokenizer,
                        image          = next(iter(val_loader))['images'][0],
                        prompt         = "<image>\n",
                        max_new_tokens = max_new_tokens,
                        temperature    = 1.5,
                        top_k          = 50,
                        device         = device
                    )
            
            print(text.replace("\n", " "))
            
    return train_losses, val_losses, track_tokens_seen
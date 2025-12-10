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

## pipline
def vision_pipeline(deep_encoder, deep_decoder, input_ids_batch, image_batch, tokenizer):
    batch_size = input_ids_batch.shape[0]

    vision_tokens = deep_encoder(image_batch)
    text_embeds = deep_decoder.token_embedding(input_ids_batch)

    image_token_id = text_to_token_ids("<image>", tokenizer)   # we will find
                                                               # <image> and replace
                                                               # with tokens from SAM
                                                               # and CLIP  
    final_embeds = []
    for batch in range(batch_size):
        image_token_mask = (image_token_id == input_ids_batch[batch])
        image_positions = torch.where(image_token_mask[batch])[0]
        img_pos = image_positions.squeeze().item()

        before = text_embeds[batch, :img_pos]
        after = text_embeds[batch, img_pos+1:]

        merged = torch.cat((before, vision_tokens[batch], after), dim = 0)
        final_embeds.append(merged)

    max_len = max(e.shape[0] for e in final_embeds)
    padded_embeds = torch.stack([
        F.pad(e, (0, 0, 0, max_len - e.shape[0]), value=50256)
        for e in final_embeds
    ])

    logits = deep_decoder(inputs_embeds = padded_embeds)

    return logits

## losses
def calc_loss_batch(pipline, deep_encoder, deep_decoder, input_batch, target_batch, image_batch, tokenizer, num_vision_tokens = 273):
    logits = pipline(deep_encoder    = deep_encoder, 
                     deep_decoder    = deep_decoder,
                     input_ids_batch = input_batch,
                     image_batch     = image_batch,
                     tokenizer       = tokenizer)
    
    batch_size, seq_len, _ = logits.shape
    # Create aligned targets: [-100 for vision tokens, actual tokens for text]
    aligned_targets = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Copy original targets starting after vision tokens
    for i in range(batch_size):
        # Skip first token (<image>) in target_ids
        text_only_targets = target_batch[i, 1:]  # Skip position 0
        text_len = (text_only_targets != -100).sum().item()
        
        # Place text starting right after vision tokens
        aligned_targets[i, num_vision_tokens:num_vision_tokens+text_len] = text_only_targets[:text_len]

    # Now compute loss on full sequence
    loss = F.cross_entropy(
        logits.reshape(-1, 50259),      # [2*293, 50259]
        aligned_targets.reshape(-1),    # [2*293]
        ignore_index=-100               # Ignores vision token positions
    )

    return loss

def calc_loss_loader(pipeline, dataloader, deep_encoder, deep_decoder, tokenizer, num_batches = None):
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    total_loss = 0
    for idx, a in tqdm(enumerate(dataloader), total = num_batches, desc = f"Calculating batch loss: {num_batches}"):
        # print(target_batch)
        if idx < num_batches:
            loss = loss = calc_loss_batch(
                                        pipline       = pipeline,
                                        deep_encoder  = deep_encoder,
                                        deep_decoder  = deep_decoder,
                                        tokenizer     = tokenizer,
                                        input_batch   = a['input_ids'],
                                        target_batch  = a['target_ids'],
                                        image_batch   = a['images']
                                    )
            total_loss += loss.item()
        else: break
    return total_loss / num_batches

## generating text
def generate_text_simple(model, tokens, max_new_tokens, context_size):
    for _ in tqdm(range(max_new_tokens), desc = f"Generating samples: {max_new_tokens}"):
        tokens = tokens[:, -context_size:] # just in case it overflows
        logits = model(tokens)
        logits = logits[:, -1, :] # last context vector
        idx_next = torch.argmax(torch.softmax(logits, dim = -1), dim = -1, keepdim=True)
        tokens = torch.cat((tokens, idx_next), dim = 1)
    return tokens

def generate_and_print_samples(model, tokenizer, device, start_context, cfg, max_new_tokens = 50):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_ids = generate_text_simple(model = model, tokens = encoded, max_new_tokens = max_new_tokens, context_size = cfg["context_length"])
    decoded = token_ids_to_text(generated_ids, tokenizer)
    print(decoded.replace("\n", " ")) # compacting
    model.train()


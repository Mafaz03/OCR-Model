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


def generate_text(deep_encoder, gpt2, projector, tokenizer, image, 
                  prompt="<image>\n", max_new_tokens=50, 
                  temperature=0.7, top_k=50, device="cpu"):
    """
    Generate text from image with temperature and top-k sampling.
    
    Args:
        deep_encoder: SAM+CLIP vision model
        gpt2: Language model
        projector: Linear(1280, 768) projection layer
        tokenizer: Modified tokenizer with <image>
        image: [3, 1024, 1024] or [1, 3, 1024, 1024] tensor
        prompt: Starting text with <image> placeholder
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
                     - 0.1-0.5: Conservative, focused
                     - 0.7-0.9: Balanced
                     - 1.0+: Creative, diverse
        top_k: Only sample from top k most likely tokens
        device: 'cpu' or 'cuda'
    
    Returns:
        generated_text: String of generated text
    """
    gpt2.eval()
    deep_encoder.eval()
    
    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, 3, 1024, 1024]
    
    image = image.to(device)
    
    # 1. Process image ONCE
    with torch.no_grad():
        vision_tokens = deep_encoder(image)           # [1, 273, 1280]
        # vision_tokens = projector(vision_tokens)      # [1, 273, 768]
    
    # 2. Tokenize prompt
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)  # [1, seq_len]
    
    # 3. Get text embeddings
    text_embeds = gpt2.token_embedding(input_ids)     # [1, seq_len, 768]
    
    # 4. Find <image> token and merge
    image_token_id = tokenizer.encode("<image>", allowed_special={"<image>", "<|endofword|>"})[0]
    image_pos = torch.where(input_ids[0] == image_token_id)[0]
    
    if len(image_pos) == 0:
        raise ValueError("Prompt must contain <image> token")
    
    img_pos = image_pos[0].item()
    
    # Merge: text_before + vision + text_after
    before = text_embeds[0, :img_pos]
    after = text_embeds[0, img_pos+1:]
    
    current_embeds = torch.cat([
        before.unsqueeze(0),
        vision_tokens,
        after.unsqueeze(0)
    ], dim=1)  # [1, 273+text_len, 768]
    
    # 5. Generate tokens with temperature and top-k sampling
    generated_ids = []
    eos_token = 50256  # <|endoftext|>
    
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc = f"Generating samples: {max_new_tokens}"):
            # Forward pass
            logits = gpt2(inputs_embeds=current_embeds)  # [1, seq_len, vocab_size]
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                # Get top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                
                # Create probability distribution from top-k
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Sample from top-k distribution
                sampled_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices[sampled_idx]
            else:
                # No top-k: sample from full distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS token
            if next_token.item() == eos_token:
                break
            
            generated_ids.append(next_token.item())
            
            # Append new token embedding
            next_embed = gpt2.token_embedding(next_token).unsqueeze(0)  # [1, 1, 768]
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
    
    # 6. Decode generated tokens
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


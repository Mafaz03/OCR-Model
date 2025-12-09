import torch
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_queries.weight = assign(
            gpt.transformer_blocks[b].attn.W_queries.weight, q_w.T)
        gpt.transformer_blocks[b].attn.W_keys.weight = assign(
            gpt.transformer_blocks[b].attn.W_keys.weight, k_w.T)
        gpt.transformer_blocks[b].attn.W_values.weight = assign(
            gpt.transformer_blocks[b].attn.W_values.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_queries.bias = assign(
            gpt.transformer_blocks[b].attn.W_queries.bias, q_b)
        gpt.transformer_blocks[b].attn.W_keys.bias = assign(
            gpt.transformer_blocks[b].attn.W_keys.bias, k_b)
        gpt.transformer_blocks[b].attn.W_values.bias = assign(
            gpt.transformer_blocks[b].attn.W_values.bias, v_b)

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def load_weights_into_gpt_modified(gpt, params):
    # CHANGED: Handle vocab size mismatch for embeddings
    pretrained_vocab_size = params['wte'].shape[0]  # 50257
    model_vocab_size = gpt.token_embedding.weight.shape[0]  # e.g., 50259
    
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    
    # CHANGED: Partial copy for token embeddings
    if model_vocab_size > pretrained_vocab_size:
        with torch.no_grad():
            gpt.token_embedding.weight[:pretrained_vocab_size].copy_(torch.tensor(params['wte']))
            torch.nn.init.normal_(gpt.token_embedding.weight[pretrained_vocab_size:], mean=0.0, std=0.02)
    else:
        gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_queries.weight = assign(
            gpt.transformer_blocks[b].attn.W_queries.weight, q_w.T)
        gpt.transformer_blocks[b].attn.W_keys.weight = assign(
            gpt.transformer_blocks[b].attn.W_keys.weight, k_w.T)
        gpt.transformer_blocks[b].attn.W_values.weight = assign(
            gpt.transformer_blocks[b].attn.W_values.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].attn.W_queries.bias = assign(
            gpt.transformer_blocks[b].attn.W_queries.bias, q_b)
        gpt.transformer_blocks[b].attn.W_keys.bias = assign(
            gpt.transformer_blocks[b].attn.W_keys.bias, k_b)
        gpt.transformer_blocks[b].attn.W_values.bias = assign(
            gpt.transformer_blocks[b].attn.W_values.bias, v_b)

        gpt.transformer_blocks[b].attn.out_proj.weight = assign(
            gpt.transformer_blocks[b].attn.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attn.out_proj.bias = assign(
            gpt.transformer_blocks[b].attn.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ff.layers[0].weight = assign(
            gpt.transformer_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[0].bias = assign(
            gpt.transformer_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ff.layers[2].weight = assign(
            gpt.transformer_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ff.layers[2].bias = assign(
            gpt.transformer_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # CHANGED: Partial copy for output head (handles vocab size mismatch)
    if model_vocab_size > pretrained_vocab_size:
        with torch.no_grad():
            gpt.out_head.weight[:pretrained_vocab_size].copy_(torch.tensor(params['wte']))
            torch.nn.init.normal_(gpt.out_head.weight[pretrained_vocab_size:], mean=0.0, std=0.02)
    else:
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

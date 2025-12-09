import torch
import numpy as np

class MutliHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False, num_heads = 2):
        super().__init__()  

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"


        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_keys    = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_queries = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_values  = torch.nn.Linear(d_in, d_out, bias = qkv_bias)
        
        self.out_proj  = torch.nn.Linear(d_out, d_out) 
        self.dropout   = torch.nn.Dropout(dropout)
        
        self.register_buffer("mask", 
                             torch.triu(torch.ones(context_length, context_length), diagonal=1)
                             )
        # print(d_out)
    def forward(self, inputs):
        batch, num_tokens, d_in = inputs.shape
        
        keys   = self.W_keys(inputs)
        querys = self.W_queries(inputs)
        values = self.W_values(inputs)

        # Splitting
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys   = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        querys = querys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)


        keys   = keys.transpose(1,2)   # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        querys = querys.transpose(1,2) # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        values = values.transpose(1,2) # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)


        attn_scores = querys @ keys.transpose(2,3)                  # self attention
        attn_scores.masked_fill_(                   
            self.mask.bool()[:num_tokens, :num_tokens], - torch.inf # Causal attention
            )
        
        # attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim = -1) # Normalizing
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)
        context_vectors = (attn_weights @ values).transpose(1,2) # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, num_heads, head_dim)
        context_vectors = context_vectors.contiguous().view(batch, num_tokens, self.d_out) # Combining it back to a concatinated form
        context_vectors = self.out_proj(context_vectors)

        return context_vectors
    



class LayerNorm(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.ones(embed_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5
    
    def forward(self, x: torch.tensor):
        mean = x.mean(dim = -1, keepdim = True)
        var  = x.var(dim = -1, keepdim = True, unbiased = False)
        # norm = (x - mean) / (torch.sqrt(var) + self.eps)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return (self.scale * norm) + self.shift
    


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class Feedforward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
                      torch.nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]),
                      GELU(),
                      torch.nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])
                    )
        
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MutliHeadAttention(
            d_in           = cfg["embedding_dim"],
            d_out          = cfg["embedding_dim"],
            context_length = cfg["context_length"],
            num_heads      = cfg["n_heads"],
            dropout        = cfg["drop_rate"],
            qkv_bias       = cfg["qkv_bias"]
        ) 

        self.ff    = Feedforward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.dropout_shortcut = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block
        x_residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        
        x = x_residual + x   

        # Feedforward block
        x_residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x_residual + x   

        return x

class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding    = torch.nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = torch.nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head   = torch.nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx, **kwargs):
        batch_size, seq_length = in_idx.shape
        toks_embeds = self.token_embedding(in_idx)
        pos_embeds  = self.position_embedding(torch.arange(0, seq_length, device = in_idx.device))

        x = toks_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits




    

import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from transformers import SamModel, SamProcessor
from transformers import CLIPVisionModel

import math


class CLIP_modified(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.ModuleDict):
        super().__init__()
        
        self.clip_model = clip_model


    def forward(self, sam_features: torch.Tensor):
        assert len(sam_features.shape) == 3, "do the flattening and transpose"

        self.batch_size = sam_features.shape[0]
        # self.sam_features = sam_features
        
        class_embeds_clip = self.clip_model.vision_model.embeddings.class_embedding.expand(self.batch_size, 1, -1) # [B, 1, 1024]
        embeddings = torch.cat((class_embeds_clip, sam_features), dim=1)   # [B, 257, 1024]
        
        # 1
        pos_embed = self.clip_model.vision_model.embeddings.position_embedding(
                self.clip_model.vision_model.embeddings.position_ids[:, :embeddings.shape[1]]
            )                                                                   # [B, 257, 1024]
        
        # 2, 3, 4
        embeddings = embeddings + pos_embed  # ADD instead of replace
        # [B, 257, 1024]

        pooled_output = self.clip_model.vision_model.post_layernorm(
                    self.clip_model.vision_model.encoder(
                        self.clip_model.vision_model.pre_layrnorm(embeddings)
                    ).last_hidden_state
                )
        
        return pooled_output


class conv_block(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = torch.nn.Conv2d(256, 512, stride = 2, kernel_size = 2)
        self.layer_2 = torch.nn.Conv2d(512, 1024, stride = 2, kernel_size = 2)
        
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

class DeepEncoder(torch.nn.Module):
    def __init__(self, sam_model: torch.nn.ModuleDict, clip_model: torch.nn.ModuleDict):
        super().__init__()
        self.sam_model = sam_model.eval()
        self.clip_model = clip_model.eval()
        self.vision_model = CLIP_modified(clip_model = clip_model)

        self.Conv_block = conv_block()
        self.projector = torch.nn.Linear(2048, 1280, bias = True)  # TODO MlpProjector
        self.image_newline = torch.nn.Parameter(torch.randn(1280))  # Learnable token
        self.view_separator = torch.nn.Parameter(torch.randn(1280))

    def __repr__(self):
        return f"SAM params: {sum(p.numel() for p in self.sam_model.parameters()):,d}\nCLIP params: {sum(p.numel() for p in self.clip_model.parameters()):,d}"
        # print(f"SAM: {sum(p.numel() for p in self.sam_model.parameters()):,d}")
        # print(f"CLIP: {sum(p.numel() for p in self.clip_model.parameters()):,d}")
    
    def forward(self, image_input):
        batch_size = image_input.shape[0]
        
        with torch.no_grad():
            local_features = self.sam_model.vision_encoder(image_input)           # [B, 256, 64, 64]
        sam_output = local_features.last_hidden_state                        # [B, 256, 64, 64]
        sam_output_conv = self.Conv_block(sam_output)                    
        sam_features = sam_output_conv.flatten(2,3).transpose(1,2)           # [B, 256, 1024]

        with torch.no_grad():
            clip_fearures = self.vision_model(sam_features = sam_features)
        
        features_total = torch.cat((clip_fearures[:, 1:], sam_features), dim = -1)

        # 1. Project to language model dimension
        projected_features = self.projector(features_total)                   # [B, 256, 1280]

        # 2. Add spatial separators (newline tokens between rows)
        h = w = int(math.sqrt(256))  # h=16, w=16
        features_2d = projected_features.view(batch_size, h, w, 1280)         # [B, 16, 16, 1280]
        
        # add new line
        
        features_with_newlines = torch.cat([
            features_2d, 
            self.image_newline[None, None, None, :].expand(batch_size, h, 1, 1280)
        ], dim=2)                                                            # [B, 16, 17, 1280]

        # flatten back
        vision_tokens = features_with_newlines.reshape(batch_size, -1, 1280) # [B, 272, 1280]

        # 3. Add separator token at the end
        
        vision_tokens = torch.cat([
            vision_tokens,
            self.view_separator[None, None, :].expand(batch_size, 1, 1280)
        ], dim=1)                                                             # [B, 273, 1280]

        return vision_tokens                                                  # [B, 273, 1280]
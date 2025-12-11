import torch
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
import tiktoken

from helper import text_to_token_ids, token_ids_to_text


class OCR_dataset(Dataset):
    def __init__(self, dataset_file_name: str, files: list, tokenizer):
        super().__init__()
        self.master_files = dataset_file_name
        self.tokenizer = tokenizer
        self.files = files  # e.g., ['1_hello.jpg', '2_world.jpg']
        
        self.transform = T.Compose([
            T.ToTensor(),                            
            T.Resize((1024, 1024)),                   
        ])
    
    def __getitem__(self, index):
        filename = self.files[index]
        
        # Extract text from filename: "1_hello.jpg" -> "hello"
        text = filename.split('_', 1)[1].split('.jpg')[0]  # FIXED: Keep underscores in label
        
        # Load image
        path = os.path.join(self.master_files, filename)
        image_np = plt.imread(path)
        image = self.transform(image_np)
        
        # FIXED: Add <image> token to input
        input_text = f"<image>\n{text}"
        input_ids = text_to_token_ids(input_text, self.tokenizer)
        
        # FIXED: Create targets for training (same as input for OCR)
        target_ids = input_ids.clone()
        
        return input_ids, target_ids, image

    def __len__(self):
        return len(self.files)
    
def ocr_collate(batch):
    """Collate with proper target shifting"""
    input_ids_list, target_ids_list, images_list = zip(*batch)
    
    # Find max length
    max_len = max(x.shape[1] if x.dim() > 1 else x.shape[0] for x in input_ids_list)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(input_ids_list, target_ids_list):
        # Handle both 1D and 2D tensors
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)  # [seq_len] → [1, seq_len]
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)  # [seq_len] → [1, seq_len]
        
        seq_len = inp.shape[1]
        pad_amount = max_len - seq_len
        
        # Pad inputs with <|endoftext|> (50256)
        padded_inp = F.pad(inp, (0, pad_amount), value=50256)
        padded_inputs.append(padded_inp)
        
        # Remove first token, add EOS at end
        shifted_tgt = torch.cat([
            tgt[:, 1:],                      # [1, seq_len-1] - remove first token
            torch.tensor([[50256]])          # [1, 1] - add EOS
        ], dim=1)
        
        # Pad targets with -100 (ignored in loss)
        padded_tgt = F.pad(shifted_tgt, (0, pad_amount), value=-100)
        padded_targets.append(padded_tgt)
    
    padded_inputs = torch.cat(padded_inputs, dim=0)
    padded_targets = torch.cat(padded_targets, dim=0)
    images = torch.stack(images_list, dim=0)
    
    return {
        'input_ids': padded_inputs,       # [batch, seq_len]
        'target_ids': padded_targets,     # [batch, seq_len]
        'images': images                  # [batch, 3, 1024, 1024]
    }

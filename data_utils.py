import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional

class TextDataset(Dataset):
    """
    Dataset for language modeling tasks
    """
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize all texts
        for text in texts:
            tokenized = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.inputs.append(tokenized.squeeze())
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = self.inputs[idx]
        # For causal language modeling, input and target are the same
        # but target is shifted by one position
        return {
            "input_ids": item[:-1],
            "labels": item[1:]
        }

def load_dataset_from_files(file_paths: List[str], tokenizer, max_length: int = 512, 
                           batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """
    Load text data from files and create a DataLoader
    """
    texts = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.extend(f.read().split('\n\n'))  # Split by paragraphs
    
    dataset = TextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_dataset_from_json(file_path: str, text_key: str, tokenizer, 
                          max_length: int = 512, batch_size: int = 16, 
                          shuffle: bool = True) -> DataLoader:
    """
    Load text data from a JSON file and create a DataLoader
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item[text_key] for item in data if text_key in item]
    dataset = TextDataset(texts, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask for padded sequences
    """
    return (input_ids != pad_token_id).float()

def create_causal_mask(seq_length: int) -> torch.Tensor:
    """
    Create causal mask for autoregressive generation
    """
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))
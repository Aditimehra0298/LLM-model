import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer

from model import LLM
from data_utils import load_dataset_from_files, load_dataset_from_json

def train(model, dataloader, optimizer, device, epochs=5):
    """
    Train the model
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save model config
    model_config = {
        "vocab_size": model.token_embedding.num_embeddings,
        "embed_dim": model.token_embedding.embedding_dim,
        "num_layers": len(model.transformer_blocks),
        "num_heads": model.transformer_blocks[0].attention.num_heads,
        "ff_dim": model.transformer_blocks[0].ff[0].out_features,
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        import json
        json.dump(model_config, f, indent=2)
    
    # Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data files or directory")
    parser.add_argument("--data_type", type=str, default="text", choices=["text", "json"], help="Type of data files")
    parser.add_argument("--json_text_key", type=str, default="text", help="Key for text in JSON data")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Pretrained tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load data
    if args.data_type == "text":
        if os.path.isdir(args.data_path):
            file_paths = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.txt')]
        else:
            file_paths = [args.data_path]
        dataloader = load_dataset_from_files(file_paths, tokenizer, args.max_seq_length, args.batch_size)
    else:  # json
        dataloader = load_dataset_from_json(args.data_path, args.json_text_key, tokenizer, args.max_seq_length, args.batch_size)
    
    # Initialize model
    model = LLM(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length
    )
    model.to(args.device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train model
    model = train(model, dataloader, optimizer, args.device, args.epochs)
    
    # Save model
    save_model(model, tokenizer, args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
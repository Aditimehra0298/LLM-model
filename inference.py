import torch
import argparse
import json
import os
from transformers import AutoTokenizer

from model import LLM

def load_model(model_dir):
    """
    Load a trained model and tokenizer
    """
    # Load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Initialize model
    model = LLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        max_seq_length=512  # Match the sequence length used in training
    )
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=50,
    temperature=1.0,
    device="cpu",
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
):
    """
    Generate text using the model
    """
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
        )
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained language model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with saved model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_dir)
    
    # Generate text
    output_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        args.max_length, 
        args.temperature, 
        args.device
    )
    
    print(f"Prompt: {args.prompt}")
    print(f"Generated text: {output_text}")

if __name__ == "__main__":
    main()
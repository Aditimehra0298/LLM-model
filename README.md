# Simple LLM Implementation

This project provides a simple implementation of a Large Language Model (LLM) based on the Transformer architecture.

## Project Structure

- `model.py`: Contains the LLM model implementation with transformer blocks
- `data_utils.py`: Utilities for data preprocessing and loading
- `train.py`: Script for training the model
- `inference.py`: Script for generating text with a trained model
- `requirements.txt`: Required dependencies

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training a Model

To train a model on your own data:

```bash
python train.py --data_path /path/to/your/data --output_dir ./output
```

### Training Options

- `--data_path`: Path to data files or directory (required)
- `--data_type`: Type of data files (`text` or `json`, default: `text`)
- `--json_text_key`: Key for text in JSON data (default: `text`)
- `--tokenizer`: Pretrained tokenizer to use (default: `gpt2`)
- `--output_dir`: Directory to save model (default: `./output`)
- `--embed_dim`: Embedding dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--ff_dim`: Feed-forward dimension (default: 2048)
- `--num_layers`: Number of transformer layers (default: 6)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 5)
- `--lr`: Learning rate (default: 5e-5)

## Generating Text

To generate text with a trained model:

```bash
python inference.py --model_dir ./output --prompt "Once upon a time"
```

### Generation Options

- `--model_dir`: Directory with saved model (required)
- `--prompt`: Text prompt to start generation (default: "Once upon a time")
- `--max_length`: Maximum number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 1.0)

## Model Architecture

The LLM implementation includes:

- Token embeddings
- Positional encodings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Dropout for regularization

## Example Usage

### Training on Text Files

```bash
python train.py --data_path ./data --data_type text --epochs 10 --output_dir ./my_model
```

### Training on JSON Data

```bash
python train.py --data_path ./data.json --data_type json --json_text_key "content" --output_dir ./my_model
```

### Text Generation

```bash
python inference.py --model_dir ./my_model --prompt "The future of AI is" --max_length 100 --temperature 0.8
```

## Interactive App (Enter Your Own Facts)

A simple Streamlit app lets you paste your own facts and generate questions or content.

### Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default, the app loads the model from `./output`. You can change the model directory in the sidebar.
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import torch as sftorch
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./llama_architecture")

from config import LlamaConfig
from model_trnsfmrs import LlamaForCausalLM, LlamaModel

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def test_model(model: LlamaForCausalLM, tokenizer):
    model.to(device)

    input_text = "Merhaba nasıl"
    output_text = "sın"
    input_tokens = tokenizer.encode(input_text)
    inputs = torch.tensor(input_tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
    
    logits = outputs.flatten(0, 1)[-1]
    print("Logits shape:", logits.shape)
    print("Logits:", logits)
    
    output_token = tokenizer.encode(input_text + output_text)[len(input_tokens)]
    print(f"Expected output token: {tokenizer.decode(output_token)}, token id: {output_token}")
    
    predicted_token = torch.softmax(logits, dim=-1).argmax().item()
    print(f"Predicted token: {tokenizer.decode(predicted_token)}, token id: {predicted_token}")
    
    crossE_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([output_token]).to(device))
    print(f"Cross-Entropy Loss: {crossE_loss.item()}")
    

def load_model(model_name, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_files = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    state_dict = sftorch.load_file(model_files, device=device)
    
    # Define model configuration
    llama_config = LlamaConfig(
    vocab_size=32768,
    emb_dim=256,
    context_length=256,
    n_heads=128,
    n_layers=20,
    n_kv_groups=64,
    hidden_dim=2048,
    )
    # define model
    model = LlamaForCausalLM(llama_config, tokenizer)
    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer

def main():
    model_name = input("Enter the model name (e.g., 'TheBloke/Llama-2-7B-Chat-GPTQ'): ")
    tokenizer_name = input("Enter the tokenizer name (e.g., 'TheBloke/Llama-2-7B-Chat-GPTQ'): ")
    model_file_path = input("Enter the model file path ('model.safetensors' by default): ") or "model.safetensors"
    model, tokenizer = load_model(model_name, tokenizer_name)
    test_model(model, tokenizer)

main()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

os.system("pip install twilio")
os.system("pip install bitsandbytes")

import sys
sys.path.append("./llama_architecture")

import torch
import json
from torch.utils.data import Dataset, DataLoader
from model_trnsfmrs import LlamaForCausalLM
from config import LlamaConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from safetensors import torch as sftorch
from huggingface_hub import hf_hub_download, upload_file
from transformers import AutoTokenizer
from bitsandbytes.optim import AdamW8bit
from twilio.rest import Client

tokenizer = AutoTokenizer.from_pretrained("aliarda/turkish-news-32k-tokenizer", use_fast=True)

device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
context_len=256

model_path = hf_hub_download(repo_id="aliarda/llama-50M-randParams", filename="llama-50M.safetensors", local_dir="/content/")
state_dict = sftorch.load_file(model_path, device=device)

llama_config = LlamaConfig(
    vocab_size=32768,
    emb_dim=256,
    context_length=context_len,
    n_heads=128,
    n_layers=20,
    n_kv_groups=64,
    hidden_dim=2048,
)

llama_model = LlamaForCausalLM(llama_config, tokenizer)
llama_model = llama_model.to(device)
llama_model.load_state_dict(state_dict)

ds = load_dataset("aliarda/turkish_books_tokenized")


shuffledDS = ds["train"].shuffle(seed=42)
halfData = shuffledDS.select(range(len(shuffledDS) // 2))


tokens_list = []
for i in tqdm(range(len(halfData))):
  tokens_list.append(2)
  tokens_list.extend(halfData[i]["tokens"])
  tokens_list.append(3)

pad_id = 1
eos_id = 3

class TextDataset(Dataset):
    def __init__(self, token_ids: list, context_length: int, stride: int):
        super().__init__()

        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]

            # truncate if the chunk is longer than context_length
            input_chunk = input_chunk[:context_length]
            target_chunk = target_chunk[:context_length]

            # pad the input and target chunks to context_length
            input_chunk += [pad_id] * (context_length - len(input_chunk))
            target_chunk += [pad_id] * (context_length - len(target_chunk))

            # truncate if the chunk is longer than context_length
            input_chunk = input_chunk[:context_length]
            target_chunk = target_chunk[:context_length]

            self.inputs.append(torch.tensor(input_chunk, dtype=torch.long))
            self.targets.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_dataloader(token_ids: list, context_len: int, stride: int, batch_size: int, shuffle: bool, device: str = "cpu"):
    dataset = TextDataset(token_ids, context_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device=device)
    )
    return dataloader

token_count4loader = int(len(tokens_list)/10)

# starting training
for i in range(1, 11):
  chunk = i
  train_dataloader = create_dataloader(tokens_list[(chunk - 1)*token_count4loader:chunk*token_count4loader], context_len, 256, 64, device)

    # twilio here

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = AdamW8bit(llama_model.parameters(), lr=1e-3)

  epoch = 2

  for epoch in range(epoch):
      total_loss = 0
      last_loss = 0
      for i, (X, Y) in enumerate(tqdm(train_dataloader)):

          X, Y = X.to(device), Y.to(device)

          pred = llama_model(X)
          loss = loss_fn(pred.flatten(0, 1), Y.flatten())
          total_loss += loss.item()
          last_loss = loss.item()

          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          del pred, loss, X, Y
          torch.cuda.empty_cache()

      average_loss = total_loss / len(train_dataloader)
      print(f"Epoch {epoch + 1} loss: {last_loss} average loss: {average_loss}")
      
      #twilio here
      
      sftorch.save_file(llama_model.state_dict(), f"llama_model_{epoch}_{chunk}.safetensors")
      upload_file(path_or_fileobj=f"llama_model_{epoch}_{chunk}.safetensors", repo_id="aliarda/llama-TB-50M-latest", path_in_repo="llama-50M-latest.safetensors", commit_message=f"upload llama_model chunk: {chunk}, epoch: {epoch}")

      # test

      trialInputs = [torch.tensor(tokenizer.encode("<bos>Libya 2011'de dönemin Devlet Başkanı Muammer Kaddafi'ye karşı")), torch.tensor(tokenizer.encode("<bos>Suriye'de 10. yılına giren iç savaş sürecinde rejimin en büyük")), torch.tensor(tokenizer.encode("<bos>Son darbe girişiminin ardından"))]

      #save a json with input and output each then save that to a file
      outputs = {
          f"examples_{chunk}_{epoch}": [
              {
                  "input": trialInputs[0],
                  "output": llama_model.generate(trialInputs[0])
              },
              {
                  "input": trialInputs[1],
                  "output": llama_model.generate(trialInputs[1])
              },
              {
                  "input": trialInputs[2],
                  "output": llama_model.generate(trialInputs[2])
              }
          ]
      }

      with open("generated_text.txt", "a") as f:
            json.dump(outputs, f)
            f.close()





from torch.utils.data import Dataset
import torch


pad_id = 5
eos_id = 6

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
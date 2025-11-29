import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextDataModule(L.LightningDataModule):
    def __init__(
        self, file_path: str, batch_size: int, num_workers: int, seq_length: int
    ):
        super().__init__()

        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length

    def setup(self, stage: str):
        if stage == "fit":
            tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/cosmo2-tokenizer", use_fast=self.num_workers == 0
            )
            tokenizer.pad_token = tokenizer.eos_token

            self.train_dataset = TextDataset(
                file_path=self.file_path,
                tokenizer=tokenizer,
                seq_length=self.seq_length,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, seq_length=256):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Read the text file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        # Get a sequence of seq_length + 1 tokens
        chunk = self.tokens[idx : idx + self.seq_length + 1]

        # Input: first seq_length tokens, Target: last seq_length tokens (shifted by 1)
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, labels

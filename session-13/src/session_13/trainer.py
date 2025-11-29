import lightning as L
import torch
import torch.nn.functional as F
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from transformers import AutoTokenizer

from .model import SmolLM2


class TextGenerationCallback(Callback):
    def __init__(self, prompts=["Once upon a time"], max_length=50, every_n_steps=500):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompts = prompts
        self.max_length = max_length
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if we should generate (every 500 steps)
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            self.generate_text(trainer, pl_module)

    @torch.no_grad()
    def generate_text(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.eval()

        rank_zero_info(f"\n{'='*80}")
        rank_zero_info(f"Generating at step {trainer.global_step}")
        rank_zero_info(f"{'='*80}")

        generated_texts = {}
        for prompt in self.prompts:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
                pl_module.device
            )

            # Generate
            generated = input_ids.clone()
            for _ in range(self.max_length):
                # Forward pass
                logits = pl_module.forward(generated)

                # Get next token (greedy)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Decode
            generated_text = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )

            rank_zero_info(f"\nPrompt: {prompt}")
            rank_zero_info(f"Generated: {generated_text}\n")

            # Store for logger (sanitize prompt for key name)
            key = prompt.replace(" ", "_")[:20]  # Short key
            generated_texts[f"generated/{key}"] = generated_text

        rank_zero_info(f"{'='*80}\n")

        # Log to tensorboard/wandb/etc as text
        if trainer.logger is not None:
            for key, text in generated_texts.items():
                if isinstance(trainer.logger, TensorBoardLogger):
                    trainer.logger.experiment.add_text(
                        key, text, global_step=trainer.global_step
                    )

        pl_module.train()


class SmolLM2Lightning(L.LightningModule):
    def __init__(
        self,
        vocab_size=49152,
        hidden_size=576,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        intermediate_size=1536,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=2000,
        max_steps=5000,
    ):
        super().__init__()

        self.model = SmolLM2(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids: Tensor):
        return self.model(input_ids)

    def training_step(self, batch: tuple[Tensor, Tensor]):
        input_ids, labels = batch
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train.loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        # Linear warmup + linear decay scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, self.warmup_steps))
            else:
                # Linear decay
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

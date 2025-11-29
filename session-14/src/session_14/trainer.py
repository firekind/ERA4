import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .model import DeepSeekMoE, SmolDeepSeek


class SmolDeepSeekLightning(L.LightningModule):
    def __init__(
        self,
        vocab_size=49152,
        hidden_size=576,
        num_hidden_layers=30,
        num_attention_heads=9,
        compression_ratio=8,
        intermediate_size=1536,
        num_routed_experts=7,
        num_shared_experts=1,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_pct=0.1,
        max_steps=11000,
    ):
        super().__init__()

        self.model = SmolDeepSeek(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            compression_ratio=compression_ratio,
            intermediate_size=intermediate_size,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            num_experts_per_tok=num_experts_per_tok,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = int(max_steps * warmup_pct)
        self.max_steps = max_steps

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids: Tensor):
        return self.model(input_ids)

    def training_step(self, batch: tuple[Tensor, Tensor]):
        input_ids, labels = batch
        logits = self.model(input_ids)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        if torch.isnan(loss) or torch.isinf(loss):
            rank_zero_warn(f"NaN/Inf loss at step {self.global_step}, skipping batch")
            return

        self.log("train.loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        # updating routing bias for all MoE layers
        for layer_idx, layer in enumerate(self.model.layers):
            moe: DeepSeekMoE | None = getattr(layer, "moe", None)
            if moe is None:
                continue

            # get expert load from the last forward pass
            expert_load = moe.last_expert_load
            if expert_load is not None:
                # updating routing bias (adaptive, loss-less load balancing)
                moe.update_bias_terms(expert_load)

                # logging expert load
                if self.global_step % 100 == 0:
                    max_load = expert_load.max().item()
                    min_load = expert_load.min().item()
                    avg_load = expert_load.mean().item()

                    self.log(
                        f"expert_load/layer_{layer_idx}/max", max_load, on_step=True
                    )
                    self.log(
                        f"expert_load/layer_{layer_idx}/min", min_load, on_step=True
                    )
                    self.log(
                        f"expert_load/layer_{layer_idx}/avg", avg_load, on_step=True
                    )

    def on_before_optimizer_step(self, optimizer: Optimizer):
        # Log gradient norms to catch explosion early
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.log("grad_norm", total_norm, on_step=True)

        # Warn if gradient norm is suspiciously high
        if total_norm > 100.0:
            rank_zero_warn(
                f"Large gradient norm: {total_norm:.2f} at step {self.global_step}"
            )

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

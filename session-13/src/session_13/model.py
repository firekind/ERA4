import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, hidden_size)

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings=2048, rope_theta=100000.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = rope_theta

        # precompute frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)

        # get frequencies
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore
        freqs = torch.outer(position_ids, inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, dim)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        num_heads=9,
        num_kv_heads=3,
        max_position_embeddings=2048,
        rope_theta=100000.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Grouped Query Attention: fewer KV hea than Q heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, num_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # reshape for multi head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_kv_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )  # (batch, num_kv_heads, seq_len, head_dim)

        # apply RoPE
        cos, sin = self.rotary_emb(seq_len, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # repeat KV heads to match Q heads (grouped query attention)
        # e.g: 3 KV heads -> 9 heads (each KV head used by 3 Q heads)
        k = k.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=1
        )  # (batch, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=1
        )  # (batch, num_heads, seq_len, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
        )

        # reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # output projection
        output = self.o_proj(attn_output)
        return output


class MLP(nn.Module):
    def __init__(self, hidden_size=576, intermediate_size=1536):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        num_heads=9,
        num_kv_heads=3,
        intermediate_size=1536,
        rms_norm_eps=1e-5,
        max_position_embeddings=2048,
        rope_theta=100000.0,
    ):
        super().__init__()

        # pre-attention norm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # attention
        self.self_attn = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, max_position_embeddings, rope_theta
        )

        # pre-mlp norm
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # mlp
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SmolLM2(nn.Module):
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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        # final layer norm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # optional separate output projection
        self.lm_head = None
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_state=False,
    ) -> torch.Tensor:
        # input_ids shape: (batch, seq_len)

        x = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)

        # pass through the transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        # final norm
        x = self.norm(x)

        # Return hidden states if requested
        if output_hidden_state:
            return x

        # project to vocabulary
        logits: torch.Tensor
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        return logits

    def _init_weights(self, module):
        std = 1.0 / 24.0  # 0.041666666666666664
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

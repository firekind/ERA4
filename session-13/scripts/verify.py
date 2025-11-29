import torch
from transformers import AutoModel

from session_13 import SmolLM2


def main():
    official_model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    official_state_dict = official_model.state_dict()

    model = SmolLM2(
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
    )

    result = model.load_state_dict(official_state_dict, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    model.eval()
    official_model.eval()

    # Create dummy input
    input_ids = torch.randint(0, 49152, (1, 10))  # (batch=1, seq_len=10)

    with torch.no_grad():
        output = model(input_ids, output_hidden_state=True)
        official_output = official_model(input_ids).last_hidden_state

        # Compare final hidden states (before lm_head)
        # Note: official model might have lm_head separate, so compare hidden states
        print(f"model output shape: {output.shape}")
        print(f"official model output shape: {official_output.shape}")

        # Check if outputs are close
        diff = torch.abs(output - official_output).max()
        print(f"max difference: {diff.item()}")

        if diff < 1e-4:
            print("✓ Success! Outputs match!")
        else:
            print("✗ Outputs don't match - architecture might be wrong")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer

from session_14 import SmolDeepSeekLightning


def main(ckpt_path: str, prompt: str):
    # Load
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

    device: str
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = SmolDeepSeekLightning.load_from_checkpoint(ckpt_path)
    model.eval().to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(100):
            logits = model.model(generated)
            next_token = torch.argmax(logits[:, -1, :], keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint to load")
    parser.add_argument("prompt", help="Prompt to use")
    args = parser.parse_args()

    main(args.ckpt_path, args.prompt)  # type: ignore

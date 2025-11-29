import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from session_14 import SmolDeepSeekLightning, TextDataModule


class CustomLightningCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("trainer.max_steps", "model.max_steps")


def main():
    CustomLightningCli(SmolDeepSeekLightning, TextDataModule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

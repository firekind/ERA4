from lightning.pytorch.cli import LightningCLI

from session_13 import SmolLM2Lightning, TextDataModule


def main():
    LightningCLI(SmolLM2Lightning, TextDataModule)


if __name__ == "__main__":
    main()

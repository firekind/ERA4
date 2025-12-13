from lightning.pytorch.cli import LightningCLI

from session_15 import OxfordPetDataModule, UNetModule


def main():
    LightningCLI(model_class=UNetModule, datamodule_class=OxfordPetDataModule)


if __name__ == "__main__":
    main()

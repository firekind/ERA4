import sys

import matplotlib.pyplot as plt
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner

from era4_mini_capstone import ImageNet, ImageNetModel


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument(
            "--find_lr",
            default=False,
            action="store_true",
            help="runs lr finder, sets the lr to the found lr (only valid when fitting the model)",
        )
        parser.add_argument(
            "--find_lr.exit_after_find",
            default=False,
            action="store_true",
            help="exits the program after finding lr (only valid when fitting the model)",
        )

    def before_fit(self):
        if not self.config["fit"].get("find_lr", False):
            return

        tuner = Tuner(self.trainer)

        print("finding and updating lr")
        lr_finder = tuner.lr_find(
            self.model,
            datamodule=self.datamodule,
            min_lr=1e-5,
            max_lr=1.0,
            num_training=100,
        )

        if lr_finder is None:
            return

        fig = lr_finder.plot(suggest=True)
        if fig is None:
            return

        if isinstance(self.trainer.logger, TensorBoardLogger):
            self.trainer.logger.experiment.add_figure("lr_finder", fig, global_step=0)

        save_dir = self.trainer.log_dir or "."
        fig.savefig(f"{save_dir}/lr_finder.png")  # type: ignore
        plt.close(fig)  # type: ignore

        if self.config["fit"].get("find_lr.exit_after_find", False):
            sys.exit(0)


def main():
    CustomLightningCLI(
        model_class=ImageNetModel,
        datamodule_class=ImageNet,
        subclass_mode_model=True,
    )


if __name__ == "__main__":
    main()

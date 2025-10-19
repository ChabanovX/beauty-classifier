import platform
import subprocess
import warnings
import os

import asyncclick as click
import dotenv

from src.infrastructure.ml_models.attractiveness import attractiveness_model


@click.group
def cli():
    pass


@cli.command
@click.option("--train", is_flag=True, default=False, help="Train the model")
@click.option("--eval", is_flag=True, default=False, help="Evaluate the model")
def attractiveness(train: bool, eval: bool):
    attractiveness_model.load()
    if train:
        attractiveness_model.train()
    if eval:
        attractiveness_model.evaluate()


def check_vm_reachable():
    ml_remote_ip = os.getenv("ML__REMOTE_IP")
    if not ml_remote_ip:
        warnings.warn("ML__REMOTE_IP not set. Skipping VM reachability check")
        return True
    param = "-c" if platform.system() == "Windows" else "-n"
    command = ["ping", param, "1", ml_remote_ip]
    result = subprocess.run(command, capture_output=True, timeout=5).returncode
    if not result == 0:
        warnings.warn(
            f"Remote VM ({os.getenv('ML__REMOTE_IP')}) not reachable."
            "Make sure it is running or try turning off VPN"
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    check_vm_reachable()
    cli()

import click

from src.infrastructure.ml_models.attractiveness.model import attractiveness_model


@click.group
def cli():
    pass


@cli.command
@click.option("--train", is_flag=True, default=False, help="Train the model")
@click.option("--eval", is_flag=True, default=False, help="Evaluate the model")
def attractiveness(train: bool, eval: bool):
    if train:
        attractiveness_model.train()
    if eval:
        attractiveness_model.evaluate()


if __name__ == "__main__":
    cli()

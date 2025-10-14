import logging.config
import click
import uvicorn

from src.interfaces.api.app import app
from src.config import config


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dev", is_flag=True, default=False, help="Run in development mode")
def run(dev: bool):
    """Run the API server."""
    config.app.dev = dev

    # Configure logging
    logging.config.dictConfig(config.logging.config)

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_config=config.logging.config,
    )


if __name__ == "__main__":
    cli()

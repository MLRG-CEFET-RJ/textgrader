#!/usr/bin/env python
import importlib
import logging
import os

import click

logger = logging.getLogger(__name__)

try:
    import colored_traceback
    colored_traceback.add_hook()
except ModuleNotFoundError:
    pass


@click.group()
def run():
    pass


@run.command("task")
@click.argument("dag")
@click.argument("task")
def task(dag, task):
    try:
        mod = importlib.import_module(f"dags.{dag}")
    except ModuleNotFoundError as ex:
        if "dag" not in ex.args[0]:
            raise
        click.secho(f"ERROR: Dag not found '{dag}'")
        logger.exception(ex)
        raise SystemExit(1)

    mod.run_dag(task)


@run.command("list-tasks")
@click.argument("dag")
def list_tasks(dag):

    dags = get_dags()

    if dag not in dags:
        click.secho(f"ERROR: DAG '{dag}' not found in dags.yaml")

    for task in dags[dag]:
        print(task)


def get_dags():
    import yaml
    from pathlib import Path

    dags_file = Path(__file__).parent / "dags.yaml"

    with dags_file.open("r") as fo:
        dags = yaml.safe_load(fo)
    return dags


@run.command("docker")
@click.argument("dag")
@click.argument("task")
@click.option("-t", "--tag", help="The tag to apply to the container.")
def docker(dag, task, tag):
    """ Run pipeline in Docker.

    Meant for testing. It automatically passes the `.env` file to the Docker run,
    and run the specified pipeline.

    If the `--tag` flag is not provided, we'll create a one based on the current user
    name.
    """

    from sh import docker

    user = os.environ["USER"]
    tag = tag or f"atena-{user}"
    docker("build", "-t", tag, ".", _fg=True)

    cwd = os.getcwd()

    run_args = ["--rm", "-i", "-v", f"{cwd}/.env:/app/.env"]
    command = ["run", "task", dag, task]
    docker("run", *run_args, tag, *command, _fg=True)


if __name__ == "__main__":
    from dags import logging

    logging.init()
    run()
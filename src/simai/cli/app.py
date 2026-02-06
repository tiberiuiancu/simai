import typer

from simai.cli.workflow import app as workflow_app
from simai.cli.simulate import app as simulate_app

app = typer.Typer(
    name="simai",
    help="SimAI — AI datacenter network simulation toolkit.",
    no_args_is_help=True,
)

app.add_typer(workflow_app, name="workflow", help="Generate training workloads.")
app.add_typer(simulate_app, name="simulate", help="Run network simulations.")


def main():
    app()

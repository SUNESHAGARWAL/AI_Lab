import typer

app = typer.Typer()


@app.command()
def status() -> None:
    """Report ingest CLI readiness. Corpus pipeline lands separately from this scaffold."""
    typer.echo("ingest CLI: skeleton only, no corpus pipeline wired up yet")


if __name__ == "__main__":
    app()

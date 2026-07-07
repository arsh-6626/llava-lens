import click
from rich.console import Console
from rich.table import Table

from llava_lens import __version__
from llava_lens.core.config import Config, get_config, set_config, load_config
from llava_lens.core.logging import setup_logging

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="llava-lens")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config_path, verbose):
    ctx.ensure_object(dict)
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level)
    config = load_config(config_path) if config_path else get_config()
    ctx.obj["config"] = config


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--prompt", "-p", default="Describe the image.", help="Prompt for the model")
@click.option("--output", "-o", type=click.Path(), help="Output HTML path")
@click.option("--layer", "-l", default=-1, type=int, help="Layer index for patch selection")
@click.option("--model", "-m", default=None, help="Model name")
@click.pass_context
def analyze(ctx, image_path, prompt, output, layer, model):
    from pathlib import Path
    from rich.progress import Progress

    config = ctx.obj["config"]
    if model:
        config.model.name = model

    setup_logging(level="INFO")
    console.print("[bold blue]LLAVA-LENS Analyze[/bold blue]")

    from llava_lens.core.registry import get_registry
    from llava_lens.analysis.pipeline import AnalysisPipeline
    from llava_lens.visualization.dashboard import DashboardGenerator

    with Progress() as progress:
        task = progress.add_task("Loading model...", total=None)
        registry = get_registry()
        model_cls = registry.get("models", "llava")
        vlm = model_cls(config)
        vlm.load()
        progress.update(task, description="Model loaded")

        pipeline = AnalysisPipeline(config, vlm)
        dashboard = DashboardGenerator(config)

        task2 = progress.add_task("Analyzing image...", total=None)
        result = pipeline.analyze(image_path, prompt, layer)
        progress.update(task2, description="Analysis complete")

    table = Table(title="Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", config.model.name)
    table.add_row("Prompt", prompt)
    table.add_row("Num Layers", str(result.logit_lens.num_layers))
    table.add_row("High Confidence Patches", str(len(result.patch_selection.high_confidence_indices)))
    table.add_row("Low Confidence Patches", str(len(result.patch_selection.low_confidence_indices)))
    table.add_row("Selected Patches", str(len(result.patch_selection.selected_indices)))
    console.print(table)

    if not output:
        output = Path(image_path).stem + "_dashboard.html"

    dashboard.generate(
        image=image_path,
        logit_lens_result=result.logit_lens,
        prompt=prompt,
        model_name=config.model.name,
        output_path=output,
    )
    console.print(f"[green]Dashboard saved to: {output}[/green]")


@cli.command()
@click.argument("image_dir", type=click.Path(exists=True))
@click.option("--prompt", "-p", default="Describe the image.", help="Prompt for the model")
@click.option("--output-dir", "-o", default="./results", help="Output directory")
@click.option("--batch-size", "-b", default=1, type=int, help="Batch size")
@click.option("--max-items", "-n", default=None, type=int, help="Max items to process")
@click.option("--model", "-m", default=None, help="Model name")
@click.pass_context
def batch(ctx, image_dir, prompt, output_dir, batch_size, max_items, model):
    from pathlib import Path
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    config = ctx.obj["config"]
    if model:
        config.model.name = model

    console.print("[bold blue]LLAVA-LENS Batch Processing[/bold blue]")

    from llava_lens.core.registry import get_registry
    from llava_lens.analysis.pipeline import AnalysisPipeline
    from llava_lens.processing.pipeline import ProcessingPipeline

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        registry = get_registry()
        model_cls = registry.get("models", "llava")
        vlm = model_cls(config)
        vlm.load()

        analysis_pipeline = AnalysisPipeline(config, vlm)
        processing_pipeline = ProcessingPipeline(config, vlm)

        def analyze_fn(image, prompt):
            return analysis_pipeline.analyze(image, prompt)

        task2 = progress.add_task("Processing images...", total=None)
        result = processing_pipeline.process_directory(
            image_dir=image_dir,
            prompt=prompt,
            analysis_fn=analyze_fn,
            output_dir=output_dir,
            batch_size=batch_size,
            max_items=max_items,
        )

    table = Table(title="Batch Processing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Images", str(result.summary["total"]))
    table.add_row("Processed", str(result.summary["processed"]))
    table.add_row("Failed", str(result.summary["failed"]))
    table.add_row("Elapsed Time", f"{result.summary['elapsed_time']:.2f}s")
    table.add_row("Items/sec", f"{result.summary['items_per_second']:.2f}")
    console.print(table)
    console.print(f"[green]Results saved to: {output_dir}[/green]")


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.pass_context
def serve(ctx, host, port, reload):
    import uvicorn

    config = ctx.obj["config"]
    console.print(f"[bold blue]Starting LLAVA-LENS server on {host}:{port}[/bold blue]")

    uvicorn.run(
        "llava_lens.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@cli.command()
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--set", "set_key", help="Set config key (e.g., model.name)")
@click.option("--value", help="Value to set")
@click.pass_context
def config_cmd(ctx, show, set_key, value):
    config = ctx.obj["config"]

    if show:
        from rich.pretty import pprint
        pprint(config.model_dump())
        return

    if set_key and value:
        keys = set_key.split(".")
        obj = config
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)
        console.print(f"[green]Set {set_key} = {value}[/green]")
    else:
        console.print("[yellow]Use --show to display config or --set <key> --value <val> to set[/yellow]")


@cli.command()
@click.option("--limit", "-l", default=20, type=int, help="Number of runs to show")
@click.pass_context
def runs(ctx, limit):
    from rich.table import Table

    config = ctx.obj["config"]
    tracker = ExperimentTracker(config)

    run_list = tracker.list_runs(limit)

    if not run_list:
        console.print("[yellow]No runs found[/yellow]")
        return

    table = Table(title="Experiment Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Start Time")

    for run in run_list:
        status_style = {
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "running": "[blue]running[/blue]",
        }.get(run.status, run.status)
        table.add_row(run.run_id, run.name or "-", status_style, run.model_name, run.start_time)
    console.print(table)


@cli.command()
@click.argument("run_id")
@click.pass_context
def run_detail(ctx, run_id):
    from rich.pretty import pprint

    config = ctx.obj["config"]
    tracker = ExperimentTracker(config)
    run = tracker.get_run(run_id)

    if not run:
        console.print(f"[red]Run {run_id} not found[/red]")
        return

    console.print(f"[bold]Run: {run.run_id}[/bold]")
    console.print(f"Name: {run.name}")
    console.print(f"Status: {run.status}")
    console.print(f"Model: {run.model_name}")

    metrics = tracker.get_run_metrics(run_id)
    if metrics:
        console.print("\n[bold]Metrics:[/bold]")
        for m in metrics:
            console.print(f"  {m.name}: {m.value}")


@cli.command()
@click.pass_context
def models(ctx):
    from rich.table import Table

    from llava_lens.core.registry import get_registry

    registry = get_registry()
    model_list = registry.list_available("models")

    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    for name in model_list:
        table.add_row(name)
    console.print(table)


@cli.command()
@click.pass_context
def cache_clear(ctx):
    config = ctx.obj["config"]
    from llava_lens.storage.cache import CacheManager
    cache = CacheManager(config)
    cache.clear()
    console.print("[green]Cache cleared[/green]")


@cli.command()
@click.pass_context
def storage_cleanup(ctx):
    config = ctx.obj["config"]
    from llava_lens.storage.local import LocalStorage
    storage = LocalStorage(config)
    storage.cleanup()
    console.print("[green]Storage cleaned up[/green]")

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from system.audit import load_run, save_run
from system.backtest.engine import run_backtest
from system.config_loader import load_stage_overrides
from system.pipeline.runner import run_pipeline
from system.product.monitor_plan import build_monitor_plan
from system.product.registry import list_products
from system.utils import ensure_dir

console = Console()


def _parse_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


@click.group()
def cli() -> None:
    """产品优先（Product-first）的约束×需求投资决策系统 v1。"""


@cli.command()
@click.option("--product", required=True, help="产品名称")
def select(product: str) -> None:
    plan = build_monitor_plan(product)
    table = Table(title=f"监控清单: {plan.product}")
    table.add_column("主题", style="cyan")
    table.add_column("约束")
    table.add_column("标的")
    for item in plan.items:
        table.add_row(item.theme, item.constraint_id, ", ".join(item.assets))
    console.print(table)


@cli.command()
@click.option("--product", required=True, help="产品名称")
@click.option("--date", "date_str", required=True, help="日期 YYYY-MM-DD")
def run(product: str, date_str: str) -> None:
    run_date = _parse_date(date_str)
    results = run_pipeline(product, run_date)
    run_record = save_run(date_str, product, results)
    table = Table(title=f"单日决策: {product} {date_str}")
    table.add_column("主题", style="cyan")
    table.add_column("意图")
    table.add_column("阶段")
    table.add_column("分数")
    table.add_column("原因")
    for result in run_record.results:
        table.add_row(result.theme, result.intent, result.stage, f"{result.score:.2f}", result.reason)
    console.print(table)
    console.print(f"运行ID: {run_record.run_id}")


@cli.command()
@click.option("--product", required=True, help="产品名称")
@click.option("--start", required=True, help="起始日期 YYYY-MM-DD")
@click.option("--end", required=True, help="结束日期 YYYY-MM-DD")
@click.option("--stage-overrides", default=None, help="阶段注入文件")
def backtest(product: str, start: str, end: str, stage_overrides: str | None) -> None:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    output_dir = Path("backtest_output") / f"{product}_{start}_{end}"
    ensure_dir(str(output_dir))
    run_backtest(product, start_date, end_date, stage_overrides, str(output_dir))
    console.print(f"回测完成，输出目录: {output_dir}")


@cli.command()
@click.option("--run-id", required=True, help="运行ID")
def replay(run_id: str) -> None:
    record = load_run(run_id)
    table = Table(title=f"回放: {record.product} {record.date}")
    table.add_column("主题", style="cyan")
    table.add_column("意图")
    table.add_column("阶段")
    table.add_column("分数")
    table.add_column("原因")
    for result in record.results:
        table.add_row(result.theme, result.intent, result.stage, f"{result.score:.2f}", result.reason)
    console.print(table)


@cli.command()
def products() -> None:
    items = list_products()
    console.print("可用产品:")
    for item in items:
        console.print(f"- {item}")


if __name__ == "__main__":
    cli()

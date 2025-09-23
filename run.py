#!/usr/bin/env python3
"""
Daily Tech Digest Agent

This script generates a daily digest of articles about LLMs, RAG, MCP, and Quantum Computing.
"""
import os
import sys
import click
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from loguru import logger
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import local modules
from crawler import WebCrawler
from processor import ContentProcessor
from ranker import ArticleRanker
from generator import DigestGenerator
from utils import load_config, setup_logging
from validator import validate_digest


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to configuration file",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging level",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str, log_level: str) -> None:
    """Daily Tech Digest Agent CLI"""
    # Setup logging
    setup_logging(level=log_level)

    # Load .env first so API keys are available
    load_dotenv()
    
    # Load configuration
    try:
        config = load_config(config_path)
        ctx.ensure_object(dict)
        ctx.obj["CONFIG"] = config
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Date for the digest (YYYY-MM-DD), defaults to today",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=None,
    help="Output directory for the digest",
)
@click.pass_context
def generate(ctx: click.Context, date: Optional[datetime], output_dir: Optional[str]) -> None:
    """Generate a new digest"""
    config = ctx.obj["CONFIG"]
    
    # Set the target date
    target_date = date.date() if date else datetime.utcnow().date()
    logger.info(f"Generating digest for {target_date}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(config["output"]["digest_dir"])
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        crawler = WebCrawler(config)
        processor = ContentProcessor(config)
        ranker = ArticleRanker(config)
        generator = DigestGenerator(config)
        
        # 1. Crawl for articles
        logger.info("Starting web crawling...")
        articles = crawler.crawl()
        
        # 2. Process and filter articles
        logger.info("Processing articles...")
        processed_articles = processor.process(articles)
        
        # 3. Rank articles
        logger.info("Ranking articles...")
        ranked_articles = ranker.rank(processed_articles)
        
        # 4. Generate digest
        logger.info("Generating digest...")
        digest_path = generator.generate(ranked_articles, target_date, output_dir)
        
        logger.success(f"Digest successfully generated at {digest_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate digest: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument("digest_file", type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def validate(ctx: click.Context, digest_file: str) -> None:
    """Validate a generated digest file"""
    ok, errors = validate_digest(digest_file)
    if ok:
        click.echo("Validation passed: all constraints satisfied ")
        sys.exit(0)
    else:
        click.echo("Validation failed with the following issues:")
        for e in errors:
            click.echo(f"- {e}")
        sys.exit(2)


if __name__ == "__main__":
    cli()

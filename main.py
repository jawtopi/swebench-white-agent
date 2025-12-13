#!/usr/bin/env python3
"""SWE-bench White Agent CLI using Strands Agents SDK.

This agent receives software engineering tasks via A2A protocol,
uses Claude to analyze the problem and generate a fix,
and returns a unified diff patch.
"""

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from white_agent.executor import create_agent, create_a2a_server, SWEBenchWhiteAgentExecutor

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="swebench-white-agent",
    help="SWE-bench White Agent - A coding agent that generates patches for GitHub issues."
)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(9002, "--port", "-p", help="Port to listen on"),
    model: str = typer.Option(
        "gpt-5-mini-2025-08-07",
        "--model", "-m",
        help="OpenAI model ID to use"
    ),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum tokens for response"),
    url: str = typer.Option(None, "--url", "-u", help="Public URL for agent card"),
):
    """Start the A2A server for the SWE-bench white agent."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting SWE-bench White Agent...")
    typer.echo(f"  Host: {host}")
    typer.echo(f"  Port: {port}")
    typer.echo(f"  Model: {model}")
    if url:
        typer.echo(f"  Public URL: {url}")
    typer.echo(f"  Agent Card: http://{host}:{port}/.well-known/agent.json")

    # Create agent and server
    agent = create_agent(
        api_key=api_key,
        model_id=model,
        max_tokens=max_tokens
    )

    server = create_a2a_server(agent, host=host, port=port, public_url=url)

    typer.echo("\nServer starting... Press Ctrl+C to stop.")

    # Start serving
    server.serve()


@app.command()
def test(
    input_file: Path = typer.Option(None, "--input", "-i", help="Input file with task description"),
    repo_path: Path = typer.Option(None, "--repo-path", "-r", help="Path to cloned repository"),
    model: str = typer.Option(
        "gpt-5-mini-2025-08-07",
        "--model", "-m",
        help="OpenAI model ID to use"
    ),
):
    """Test the agent with a sample input (without starting the server)."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.echo("Error: OPENAI_API_KEY environment variable not set", err=True)
        raise typer.Exit(1)

    # Sample task if no input provided
    if input_file:
        user_input = input_file.read_text()
    else:
        # Use exact format from green agent (includes repo_url)
        user_input = """You are a software engineer tasked with fixing a bug in an open source project.

<task_id>django__django-11099</task_id>

<repository>django/django</repository>

<repo_url>https://github.com/django/django</repo_url>

<base_commit>HEAD</base_commit>

<problem_statement>
UsernameValidator allows trailing newline in usernames

Description
ASCIIUsernameValidator and UnicodeUsernameValidator use the regex
r'^[\\w.@+-]+$'
The use of $ instead of \\Z allows a trailing newline in the username.

Should be:
r'^[\\w.@+-]+\\Z'
</problem_statement>

<hints>
Look at django/contrib/auth/validators.py
</hints>

Please analyze this issue and provide a fix as a unified diff patch.
Wrap your patch in <patch>...</patch> tags."""

    typer.echo("Testing agent with input:")
    typer.echo("-" * 40)
    typer.echo(user_input[:500] + "..." if len(user_input) > 500 else user_input)
    typer.echo("-" * 40)

    # Resolve repo path (if provided, skips auto-cloning)
    effective_repo_path = str(repo_path) if repo_path else None
    if effective_repo_path:
        typer.echo(f"\nUsing provided repo path: {effective_repo_path}")
    else:
        typer.echo("\nWill clone repository from repo_url if provided...")

    typer.echo("Processing...\n")

    executor = SWEBenchWhiteAgentExecutor(api_key=api_key, model_id=model)
    response = executor.execute(user_input, repo_path=effective_repo_path, cleanup=False)

    typer.echo("\nResponse:")
    typer.echo("=" * 40)
    typer.echo(response)


@app.command()
def version():
    """Show version information."""
    from white_agent import __version__
    typer.echo(f"SWE-bench White Agent v{__version__}")
    typer.echo("Built with Strands Agents SDK")


if __name__ == "__main__":
    app()

# SWE-bench White Agent

A minimal white agent implementation for [SWE-bench](https://www.swebench.com/) evaluation, built with the **Strands Agents SDK**. This agent receives GitHub issues via the A2A protocol, explores repositories using tools, and generates unified diff patches.

## Overview

This white agent is designed to work with the [AgentBeats](https://v2.agentbeats.org) evaluation platform. It:

1. Receives tasks from a green agent (orchestrator) containing GitHub issue details
2. Clones the repository at the specified commit
3. Uses tools to explore and understand the codebase
4. Generates a unified diff patch to fix the issue

## Architecture

```
Green Agent (Orchestrator)
        |
        v  (A2A Protocol)
+-------------------+
|   White Agent     |
|-------------------|
|  - read_file      |
|  - search_code    |  --> Repository
|  - find_files     |
|  - list_directory |
|  - get_file_info  |
+-------------------+
        |
        v
   <patch>...</patch>
```

**Model**: `gpt-4o-mini` (configurable)

**Framework**: [Strands Agents SDK](https://strandsagents.com/) with built-in memory and context management

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Run Locally

```bash
# Start the A2A server
python main.py serve --port 9002

# Or use the run script
./run.sh
```

### 4. Test the Agent

```bash
# Test with a sample Django issue (will auto-clone the repo)
python main.py test
```

## Deployment on AgentBeats

### Option 1: Using AgentBeats Controller (Recommended)

```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export CLOUDRUN_HOST=your-deployment-url.com
export HTTPS_ENABLED=true

# Run with controller
agentbeats run_ctrl
```

### Option 2: Cloud Run Deployment

```bash
# Deploy to Google Cloud Run
./deploy-cloudrun.sh
```

### Option 3: Railway / Other PaaS

The `Procfile` is configured for platforms like Railway:
```
web: agentbeats run_ctrl
```

After deployment, register your agent on [v2.agentbeats.org](https://v2.agentbeats.org) as an "Assessee (White)" agent.

## CLI Commands

```bash
# Start A2A server
python main.py serve --host 0.0.0.0 --port 9002

# Start with custom model
python main.py serve --model gpt-4o-mini --port 9002

# Test with sample task
python main.py test

# Test with specific repository (skips cloning)
python main.py test --repo-path /path/to/django

# Show version
python main.py version
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `HOST` | Server bind address | `0.0.0.0` |
| `AGENT_PORT` | Server port | `9002` |
| `REPOS_DIR` | Directory for cloned repos | `/tmp/swebench_repos` |
| `CLOUDRUN_HOST` | Public hostname for agent card | - |
| `HTTPS_ENABLED` | Use HTTPS URLs in agent card | `false` |

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `search_code` | Search for patterns using grep |
| `find_files` | Find files by name |
| `list_directory` | List directory tree |
| `get_file_info` | Get file size and line count |

## Input/Output Format

**Input** (from green agent):
```xml
<task_id>django__django-11099</task_id>
<repository>django/django</repository>
<repo_url>https://github.com/django/django</repo_url>
<base_commit>abc123...</base_commit>
<problem_statement>
Description of the bug...
</problem_statement>
```

**Output**:
```
<patch>
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,6 +10,7 @@
 context line
-old line
+new line
 context line
</patch>
```

## Project Structure

```
swebench-white-agent/
├── main.py                 # CLI entry point
├── run.sh                  # AgentBeats startup script
├── Procfile                # PaaS deployment config
├── Dockerfile              # Container config
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── src/white_agent/
    ├── __init__.py
    └── executor.py         # Agent implementation
```

## Reproducing Evaluation Results

1. Deploy the agent to a publicly accessible URL
2. Register on AgentBeats as a white agent
3. Run evaluation batches (we tested with batches of 10 tasks)
4. Results are reported by the AgentBeats platform

**Expected performance**: ~10-20% resolve rate on SWE-bench Verified tasks with `gpt-4o-mini`.

## Built With

- [Strands Agents SDK](https://strandsagents.com/) - Agent framework
- [OpenAI GPT](https://openai.com/) - Language model
- [A2A Protocol](https://github.com/google/a2a) - Agent communication
- [AgentBeats](https://v2.agentbeats.org) - Evaluation platform

## License

MIT

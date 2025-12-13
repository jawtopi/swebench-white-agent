# SWE-bench White Agent

A SWE-bench white agent built with the **Strands Agents SDK** that analyzes repositories and generates unified diff patches for GitHub issues.

## Features

- **Real code analysis** - Uses tools to read and search the actual repository
- Receives software engineering tasks via A2A protocol
- Uses OpenAI GPT models to analyze problems and generate fixes
- Returns unified diff patches wrapped in `<patch>...</patch>` tags
- Designed for evaluation by the AgentBeats platform

## Tools Available to the Agent

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `list_directory` | List directory contents (tree view) |
| `search_code` | Search for patterns in code (grep) |
| `find_files` | Find files by name |
| `get_file_info` | Get file size and line count |

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run the Server

```bash
python main.py serve --port 9002
```

Or use the run script:

```bash
./run.sh
```

### 3. Test the Agent

```bash
python main.py test
```

## How It Works

1. **Receives task** with XML tags including `<repo_url>` and `<base_commit>`
2. **Auto-clones** the repository and checks out the correct commit
3. **Explores** the codebase using tools (find, grep, read)
4. **Generates** a unified diff patch based on actual file contents
5. **Returns** the patch wrapped in `<patch>...</patch>` tags

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `HOST` | Server host | `0.0.0.0` |
| `AGENT_PORT` | Server port | `9002` |
| `REPOS_DIR` | Directory for auto-cloned repos | `/tmp/swebench_repos` |
| `REPO_PATH` | Pre-cloned repo path (skips cloning) | - |

## CLI Commands

```bash
# Start the A2A server
python main.py serve --host 0.0.0.0 --port 9002

# Test with sample input (requires repo to be cloned)
python main.py test --repo-path /path/to/django

# Test with file input
python main.py test --input task.txt --repo-path /path/to/repo

# Show version
python main.py version
```

## Input Format

The agent expects messages in this format (from the green agent):

```xml
You are a software engineer tasked with fixing a bug in an open source project.

<task_id>django__django-11099</task_id>

<repository>django/django</repository>

<repo_url>https://github.com/django/django</repo_url>

<base_commit>d4df5e1b0b1c643fe0fc5a9a8a5fcd7e5a5f5e5a</base_commit>

<problem_statement>
UsernameValidator allows trailing newline in usernames...
</problem_statement>

<hints>
[Optional hints if available]
</hints>

Please analyze this issue and provide a fix as a unified diff patch.
Wrap your patch in <patch>...</patch> tags.
```

The agent will **automatically clone** the repository from `<repo_url>` and checkout `<base_commit>` before analyzing.

## Output Format

The agent returns a response containing a unified diff patch:

```
<patch>
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
├── main.py                              # CLI entry point
├── run.sh                               # AgentBeats controller script
├── Procfile                             # Railway deployment
├── Dockerfile                           # Docker container config
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Project configuration
├── .env.example                         # Environment template
└── src/
    └── white_agent/
        ├── __init__.py
        └── executor.py                  # Strands Agent + A2A server
```

## Deploy on Railway

1. Push code to GitHub
2. Create new project on [railway.app](https://railway.app)
3. Add environment variables:
   - `CLOUDRUN_HOST=<your-app>.up.railway.app`
   - `HTTPS_ENABLED=true`
   - `OPENAI_API_KEY=sk-...`
4. Deploy and get URL
5. Register on [v2.agentbeats.org](https://v2.agentbeats.org) as "Assessee (White)" agent

## Local Development with AgentBeats Controller

```bash
# Run with controller (recommended)
PORT=9002 CLOUDRUN_HOST=<your-tunnel>.trycloudflare.com HTTPS_ENABLED=true agentbeats run_ctrl

# Start cloudflare tunnel in another terminal
cloudflared tunnel --url http://localhost:9002
```

## Built With

- [Strands Agents SDK](https://strandsagents.com/) - Agent framework by AWS
- [OpenAI GPT](https://openai.com/) - LLM for code analysis
- [A2A Protocol](https://github.com/google/a2a) - Agent-to-Agent communication
- [AgentBeats](https://v2.agentbeats.org) - Agent evaluation platform

## License

MIT

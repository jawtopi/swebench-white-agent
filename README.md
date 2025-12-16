# SWE-bench White Agent

A minimal, modular white agent for [SWE-bench](https://www.swebench.com/) evaluation built with the **Strands Agents SDK**.

---

## What is This?

This agent receives GitHub issues and generates code patches to fix bugs in real open-source repositories.

```
Input:  GitHub issue description + repository
Output: Unified diff patch that fixes the bug
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GREEN AGENT                          │
│                   (Orchestrator)                        │
└─────────────────────┬───────────────────────────────────┘
                      │ A2A Protocol
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    WHITE AGENT                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Strands Agents SDK                   │  │
│  │  ┌─────────────┐    ┌──────────────────────────┐  │  │
│  │  │ GPT-5-mini  │    │        6 Tools           │  │  │
│  │  │   (LLM)     │    │  • read_file             │  │  │
│  │  └─────────────┘    │  • search_code           │  │  │
│  │                     │  • find_files            │  │  │
│  │                     │  • list_directory        │  │  │
│  │                     │  • get_file_info         │  │  │
│  │                     │  • validate_patch        │  │  │
│  │                     └──────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              CLONED GIT REPOSITORY                      │
│         (Django, scikit-learn, sympy, etc.)             │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
                 <patch>...</patch>
```

---

## Decision-Making Pipeline

```
1. PARSE      →  Extract task_id, repo_url, commit, problem_statement
2. CLONE      →  Clone repository at specified commit
3. EXPLORE    →  Use search_code, find_files to locate relevant code
4. READ       →  Use read_file to understand the buggy code
5. GENERATE   →  Create unified diff patch
6. VALIDATE   →  Use validate_patch to verify it applies cleanly
7. SUBMIT     →  Return patch wrapped in <patch>...</patch> tags
```

---

## Tools Available

| Tool | Description | Example Use |
|------|-------------|-------------|
| `read_file` | Read file with line numbers | Find exact line numbers for patch |
| `search_code` | Grep patterns in codebase | Locate where a function is defined |
| `find_files` | Find files by name | Find `validators.py` in Django |
| `list_directory` | Show directory tree | Understand project structure |
| `get_file_info` | Get file metadata | Check file size before reading |
| `validate_patch` | Dry-run git apply | Verify patch before submitting |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/jawtopi/swebench-white-agent.git
cd swebench-white-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add:
# OPENAI_API_KEY=sk-...
```

### 3. Run Locally

```bash
# Start the A2A server
python main.py serve --port 9002
```

### 4. Test with Sample Task

```bash
# Test with a Django issue (auto-clones repo)
python main.py test
```

---

## Deployment on AgentBeats

### Option A: Cloud Run (Recommended)

```bash
# Deploy to Google Cloud Run
gcloud run deploy swebench-white-agent \
  --source . \
  --region us-central1 \
  --set-env-vars "OPENAI_API_KEY=sk-..." \
  --set-env-vars "CLOUDRUN_HOST=your-url.run.app" \
  --set-env-vars "HTTPS_ENABLED=true" \
  --min-instances 1
```

### Option B: Railway

```bash
# Push to GitHub, connect Railway, add env vars
# Procfile uses: agentbeats run_ctrl
```

### Option C: Local with Tunnel

```bash
# Terminal 1: Start agent
cloudflared tunnel --url http://localhost:8081

# Terminal 2: Create tunnel
export CLOUDRUN_HOST=<your cloudflared tunnel domain name>
export HTTPS_ENABLED=true
agentbeats run_ctrl
```

After deployment, register on [v2.agentbeats.org](https://v2.agentbeats.org) as "Assessee (White)" agent.

---

## Input/Output Format

### Input (from Green Agent)

```xml
<task_id>django__django-11099</task_id>
<repository>django/django</repository>
<repo_url>https://github.com/django/django</repo_url>
<base_commit>abc123...</base_commit>
<problem_statement>
UsernameValidator allows trailing newline in usernames.
The regex uses $ instead of \Z which allows trailing newlines.
</problem_statement>
```

### Output (to Green Agent)

```diff
<patch>
diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py
--- a/django/contrib/auth/validators.py
+++ b/django/contrib/auth/validators.py
@@ -8,7 +8,7 @@

 @deconstructible
 class ASCIIUsernameValidator(validators.RegexValidator):
-    regex = r'^[\w.@+-]+$'
+    regex = r'^[\w.@+-]+\Z'
     message = _(
         'Enter a valid username.'
     )
</patch>
```

---

## Reproducing Evaluation Results

### Step 1: Deploy White Agent

```bash
# Ensure white agent is running and accessible
curl https://your-white-agent-url/.well-known/agent-card.json
```

### Step 2: Run Green Agent Assessment

```bash
# On your green agent machine
cd swebench-green-agent

# Run evaluation (use max_workers=1 for stability)
python main.py evaluate \
  --white-agent-url https://your-white-agent-url \
  --sample-size 10 \
  --max-workers 1
```

### Step 3: View Results

```
============================================================
GREEN AGENT: EVALUATION COMPLETE
============================================================
Dataset: SWE-bench Verified
Total tasks: 10
Resolved: 1/10 (10.0%)
Failed: 7
Errors: 2
============================================================
```

---

## Expected Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Resolve Rate** | 10-20% | Patches that fix the issue |
| **Patch Generation** | 80-90% | Valid unified diff produced |
| **Avg Tool Calls** | 20-50 | Per task exploration |
| **Avg Time** | 60-90s | Per task completion |

**Model**: GPT-5-mini (cost-efficient, not SOTA)

---

## Project Structure

```
swebench-white-agent/
├── main.py                     # CLI entry point
│   └── serve                   # Start A2A server
│   └── test                    # Test with sample task
│   └── version                 # Show version
│
├── src/white_agent/
│   ├── __init__.py
│   └── executor.py             # Core implementation
│       ├── Tools (6)           # read_file, search_code, etc.
│       ├── SYSTEM_PROMPT       # Agent instructions
│       ├── SWEBenchA2AExecutor # Task execution logic
│       └── A2AServerWithStatus # AgentBeats compatibility
│
├── run.sh                      # Startup script
├── Procfile                    # web: agentbeats run_ctrl
├── Dockerfile                  # Container config
├── requirements.txt            # Dependencies
└── .env.example                # Environment template
```

---

## Key Design Decisions

### 1. Fresh Agent Per Task
Each task gets a new agent instance - no conversation history leakage between tasks.

### 2. Auto Repository Cloning
Agent clones the exact commit specified, ensuring reproducible environments.

### 3. Validate Before Submit
`validate_patch` tool runs `git apply --check` to catch malformed patches.

### 4. Strands SDK Benefits
- Built-in memory management
- Tool execution framework
- A2A protocol support

---

## CLI Reference

```bash
# Start server
python main.py serve --host 0.0.0.0 --port 9002

# Start with custom model
python main.py serve --model gpt-5-mini --port 9002

# Test locally
python main.py test

# Test with specific repo
python main.py test --repo-path /path/to/django

# Show version
python main.py version
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `HOST` | No | Server bind address (default: 0.0.0.0) |
| `AGENT_PORT` | No | Server port (default: 9002) |
| `REPOS_DIR` | No | Clone directory (default: /tmp/swebench_repos) |
| `CLOUDRUN_HOST` | No | Public hostname for agent card |
| `HTTPS_ENABLED` | No | Use HTTPS in agent card URLs |

---

## Limitations & Future Work

### Current Limitations
- Uses GPT-5-mini
- No bash execution for running tests

### Potential Improvements
- Add `run_tests` tool for verification
- Use stronger model for complex tasks
- Add planning/reasoning steps
- Containerized execution environment

---

## Built With

- [Strands Agents SDK](https://strandsagents.com/) - Agent framework by AWS
- [OpenAI](https://openai.com/) - Language model
- [A2A Protocol](https://github.com/google/a2a) - Agent communication
- [AgentBeats](https://v2.agentbeats.org) - Evaluation platform

---

## Repository

**GitHub**: https://github.com/jawtopi/swebench-white-agent

**Branch**: `main`

---

## License

MIT

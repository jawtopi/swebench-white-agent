"""SWE-bench White Agent Executor using Strands Agents SDK."""

import gc
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.multiagent.a2a import A2AServer
from strands.handlers.callback_handler import PrintingCallbackHandler

logger = logging.getLogger(__name__)


# Global variable for repository path (set when processing a task)
REPO_PATH: Optional[str] = None

# Directory to store cloned repos
REPOS_DIR: str = os.getenv("REPOS_DIR", "/tmp/swebench_repos")


def set_repo_path(path: str) -> None:
    """Set the repository path for tools to use."""
    global REPO_PATH
    REPO_PATH = path


def get_repo_path() -> str:
    """Get the repository path, with fallback to environment variable."""
    global REPO_PATH
    if REPO_PATH:
        return REPO_PATH
    return os.getenv("REPO_PATH", "/repo")


def clone_repository(repo_url: str, base_commit: str, task_id: str) -> str:
    """Clone a repository and checkout the specified commit.

    Args:
        repo_url: GitHub URL (e.g., https://github.com/django/django)
        base_commit: Commit hash to checkout
        task_id: Task ID for naming the directory

    Returns:
        Path to the cloned repository
    """
    # Create repos directory if it doesn't exist
    os.makedirs(REPOS_DIR, exist_ok=True)

    # Create a unique directory name based on task_id
    # Replace problematic characters
    safe_task_id = task_id.replace("/", "_").replace("__", "_")
    repo_dir = Path(REPOS_DIR) / safe_task_id

    # If directory exists and has the right commit, reuse it
    if repo_dir.exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            current_commit = result.stdout.strip()
            if current_commit.startswith(base_commit) or base_commit.startswith(current_commit):
                print(f"Reusing existing repo at {repo_dir}")
                return str(repo_dir)
        except Exception:
            pass
        # Remove and re-clone if commit doesn't match
        shutil.rmtree(repo_dir, ignore_errors=True)

    print(f"Cloning {repo_url} to {repo_dir}...")

    try:
        # Clone the repository (shallow clone for speed)
        subprocess.run(
            ["git", "clone", "--depth", "100", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for clone
        )

        # Try to checkout specific commit if not HEAD
        if base_commit and base_commit.upper() != "HEAD":
            try:
                # Try to checkout directly first
                result = subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    # Fetch more history and retry
                    subprocess.run(
                        ["git", "fetch", "--depth", "1", "origin", base_commit],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    subprocess.run(
                        ["git", "checkout", base_commit],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                print(f"Checked out {base_commit[:12]}")
            except Exception as e:
                print(f"Warning: Could not checkout {base_commit}, using default branch: {e}")

        print(f"Successfully cloned to {repo_dir}")
        return str(repo_dir)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"Failed to clone repository: {error_msg}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Repository clone timed out")


def cleanup_repository(repo_path: str) -> None:
    """Remove a cloned repository to free up space."""
    try:
        if repo_path and Path(repo_path).exists():
            shutil.rmtree(repo_path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup {repo_path}: {e}")


@tool
def read_file(file_path: str, start_line: int = 1, end_line: int = None) -> str:
    """Read contents of a file from the repository.

    Args:
        file_path: Path to the file relative to repository root (e.g., 'django/contrib/auth/validators.py')
        start_line: Starting line number (1-indexed, default: 1)
        end_line: Ending line number (inclusive, default: read to end)

    Returns:
        File contents with line numbers, or error message if file not found.
    """
    repo_path = get_repo_path()
    full_path = Path(repo_path) / file_path

    try:
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        if not full_path.is_file():
            return f"Error: {file_path} is not a file"

        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Apply line range
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        selected_lines = lines[start_idx:end_idx]

        # Format with line numbers
        result = []
        for i, line in enumerate(selected_lines, start=start_line):
            result.append(f"{i:4d} | {line.rstrip()}")

        return '\n'.join(result) if result else "File is empty"

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_directory(dir_path: str = ".", max_depth: int = 2) -> str:
    """List contents of a directory in the repository.

    Args:
        dir_path: Path to directory relative to repository root (default: root)
        max_depth: Maximum depth to recurse (default: 2)

    Returns:
        Directory tree listing.
    """
    repo_path = get_repo_path()
    full_path = Path(repo_path) / dir_path

    try:
        if not full_path.exists():
            return f"Error: Directory not found: {dir_path}"

        if not full_path.is_dir():
            return f"Error: {dir_path} is not a directory"

        result = []

        def walk_dir(path: Path, prefix: str, depth: int):
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return

            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "

                if item.is_dir():
                    result.append(f"{prefix}{connector}{item.name}/")
                    extension = "    " if is_last else "│   "
                    walk_dir(item, prefix + extension, depth + 1)
                else:
                    result.append(f"{prefix}{connector}{item.name}")

        result.append(f"{dir_path}/")
        walk_dir(full_path, "", 1)

        return '\n'.join(result[:500])  # Limit output

    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def search_code(pattern: str, file_pattern: str = "*.py", max_results: int = 50) -> str:
    """Search for a pattern in repository files using grep.

    Args:
        pattern: Regex pattern to search for
        file_pattern: Glob pattern for files to search (default: '*.py')
        max_results: Maximum number of results to return (default: 50)

    Returns:
        Matching lines with file paths and line numbers.
    """
    repo_path = get_repo_path()

    try:
        # Use grep for searching
        cmd = [
            'grep', '-rn', '--include', file_pattern,
            '-E', pattern, repo_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        lines = result.stdout.strip().split('\n')[:max_results]

        # Clean up paths to be relative
        cleaned = []
        for line in lines:
            if line:
                # Remove repo_path prefix
                cleaned_line = line.replace(repo_path + '/', '')
                cleaned.append(cleaned_line)

        if not cleaned:
            return f"No matches found for pattern: {pattern}"

        return '\n'.join(cleaned)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def find_files(filename: str, max_results: int = 20) -> str:
    """Find files by name in the repository.

    Args:
        filename: Filename or pattern to search for (e.g., 'validators.py' or '*.py')
        max_results: Maximum number of results (default: 20)

    Returns:
        List of matching file paths.
    """
    repo_path = get_repo_path()

    try:
        cmd = ['find', repo_path, '-name', filename, '-type', 'f']

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        lines = result.stdout.strip().split('\n')[:max_results]

        # Clean up paths to be relative
        cleaned = []
        for line in lines:
            if line:
                cleaned_line = line.replace(repo_path + '/', '')
                cleaned.append(cleaned_line)

        if not cleaned:
            return f"No files found matching: {filename}"

        return '\n'.join(cleaned)

    except subprocess.TimeoutExpired:
        return "Error: Search timed out"
    except Exception as e:
        return f"Error finding files: {str(e)}"


@tool
def get_file_info(file_path: str) -> str:
    """Get information about a file (size, line count).

    Args:
        file_path: Path to file relative to repository root

    Returns:
        File information including size and line count.
    """
    repo_path = get_repo_path()
    full_path = Path(repo_path) / file_path

    try:
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        stat = full_path.stat()
        size = stat.st_size

        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            line_count = sum(1 for _ in f)

        return f"File: {file_path}\nSize: {size} bytes\nLines: {line_count}"

    except Exception as e:
        return f"Error getting file info: {str(e)}"


SYSTEM_PROMPT = """You are an expert software engineer. Your task is to fix bugs in open source repositories by analyzing code and generating patches.

You have access to tools to explore and read the repository:
- read_file: Read file contents with line numbers
- list_directory: List directory contents
- search_code: Search for patterns in code (grep)
- find_files: Find files by name
- get_file_info: Get file size and line count

WORKFLOW:
1. Analyze the problem statement to understand what needs to be fixed
2. Use tools to explore the repository and find relevant files
3. Read the specific files that need to be modified
4. Understand the existing code and identify the bug
5. Generate a unified diff patch to fix the issue

TASK FORMAT:
You will receive messages with XML tags:
- <task_id>: The SWE-bench task identifier
- <repository>: The GitHub repository (e.g., django/django)
- <base_commit>: The commit hash
- <problem_statement>: The GitHub issue description
- <hints>: Optional hints

PATCH FORMAT (MUST follow exactly):
<patch>
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line_number,count +line_number,count @@
 context line (unchanged, starts with space)
-line to remove (starts with minus)
+line to add (starts with plus)
 context line (unchanged, starts with space)
</patch>

RULES:
- ALWAYS use tools to read the actual code before generating a patch
- Include 3 lines of context before/after changes
- File paths must be relative to repo root (no leading /)
- Line numbers in @@ must be accurate based on the file you read
- ALWAYS wrap your final patch in <patch>...</patch> tags
- Only modify what's necessary to fix the issue"""


def parse_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from an XML tag."""
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_task_input(user_input: str) -> dict:
    """Parse the incoming task to extract all fields from the green agent message."""
    result = {
        "task_id": parse_xml_tag(user_input, "task_id"),
        "repo": parse_xml_tag(user_input, "repository"),
        "repo_url": parse_xml_tag(user_input, "repo_url"),  # GitHub URL to clone
        "base_commit": parse_xml_tag(user_input, "base_commit"),
        "problem_statement": parse_xml_tag(user_input, "problem_statement"),
        "hints": parse_xml_tag(user_input, "hints"),
        "repo_path": parse_xml_tag(user_input, "repo_path"),  # Optional: pre-cloned path
        "raw_input": user_input
    }

    # Fallback: if no XML tags found, use the whole input as problem statement
    if not result["problem_statement"]:
        result["problem_statement"] = user_input

    # Construct repo_url from repository if not provided
    if not result["repo_url"] and result["repo"]:
        result["repo_url"] = f"https://github.com/{result['repo']}"

    return result


def extract_patch(response: str) -> str:
    """Extract the patch from the response, ensuring it's wrapped in <patch> tags."""
    # Check if already has patch tags
    patch_match = re.search(r'<patch>(.*?)</patch>', response, re.DOTALL)
    if patch_match:
        return response

    # Try to find diff content and wrap it
    diff_patterns = [
        r'(---\s+a/.*?(?=\n(?:---|$|\Z)))',  # Standard unified diff
        r'(diff --git.*?(?=\ndiff --git|\Z))',  # Git diff format
    ]

    for pattern in diff_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            patch_content = '\n'.join(matches)
            return f"{response}\n\n<patch>\n{patch_content}\n</patch>"

    # If no diff found, wrap the entire response
    return f"<patch>\n{response}\n</patch>"


class SWEBenchA2AExecutor(AgentExecutor):
    """Custom A2A executor that creates a fresh agent for each task.

    This executor ensures:
    1. Each task gets a fresh agent instance (no shared conversation history)
    2. Repository is cloned for each task
    3. Repository is cleaned up after each task
    4. Memory is properly managed between tasks
    5. Concurrency is limited to prevent resource exhaustion
    """

    # Limit concurrent tasks to prevent resource exhaustion
    # Set to 5 to allow 4 workers + buffer
    MAX_CONCURRENT_TASKS = 5

    # Cleanup all old repos every N tasks to prevent disk exhaustion
    CLEANUP_INTERVAL = 10

    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-5-mini-2025-08-07",
        max_tokens: int = 4096
    ):
        """Initialize the executor with model configuration.

        Args:
            api_key: OpenAI API key
            model_id: Model ID to use
            max_tokens: Maximum tokens for response
        """
        import asyncio
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._task_count = 0
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Create ONE shared OpenAI model to reuse the connection pool
        # This prevents connection exhaustion from creating new clients per task
        self._shared_model = OpenAIModel(
            client_args={"api_key": self.api_key},
            model_id=self.model_id,
            params={"max_completion_tokens": self.max_tokens}
        )
        logger.info("Created shared OpenAI model instance")

    def _get_semaphore(self) -> 'asyncio.Semaphore':
        """Get or create the semaphore for limiting concurrency."""
        import asyncio
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_TASKS)
        return self._semaphore

    def _create_fresh_agent(self) -> Agent:
        """Create a fresh agent instance with no conversation history.

        Reuses the shared OpenAI model to avoid creating new HTTP connections,
        but creates a fresh Agent to ensure no conversation history is shared.
        """
        # Create fresh agent with SHARED model (reuses connection pool)
        agent = Agent(
            name="SWE-bench White Agent",
            description="A coding agent that analyzes GitHub issues and generates unified diff patches to fix bugs.",
            model=self._shared_model,  # Reuse shared model
            system_prompt=SYSTEM_PROMPT,
            tools=[read_file, list_directory, search_code, find_files, get_file_info],
            callback_handler=PrintingCallbackHandler()
        )

        return agent

    def _extract_text_from_parts(self, parts: list[Part]) -> str:
        """Extract text content from A2A message parts."""
        texts = []
        for part in parts:
            if isinstance(part.root, TextPart):
                texts.append(part.root.text)
        return "\n".join(texts)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a SWE-bench task with fresh agent and proper cleanup.

        Args:
            context: A2A request context
            event_queue: Event queue for responses
        """
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # Track task for logging
        self._task_count += 1
        task_num = self._task_count

        # Limit concurrency to prevent resource exhaustion
        semaphore = self._get_semaphore()
        logger.info(f"Task #{task_num} waiting for semaphore (max {self.MAX_CONCURRENT_TASKS} concurrent)")

        async with semaphore:
            await self._execute_task(context, updater, task_num)

    def _log_memory_usage(self, label: str) -> None:
        """Log current memory usage for debugging."""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"[Memory] {label}: {mem_mb:.1f} MB")
        except ImportError:
            pass  # psutil not available

    async def _execute_task(
        self,
        context: RequestContext,
        updater: TaskUpdater,
        task_num: int
    ) -> None:
        """Execute a single task (called within semaphore)."""
        self._log_memory_usage(f"Task #{task_num} start")
        logger.info(f"Starting task #{task_num}")

        # Extract the message text
        user_input = ""
        if context.message and hasattr(context.message, "parts"):
            user_input = self._extract_text_from_parts(context.message.parts)

        if not user_input:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message("No input provided", updater.context_id, updater.task_id)
            )
            return

        # Parse task info
        task_info = parse_task_input(user_input)
        task_id = task_info.get('task_id', f'task_{task_num}')
        repo_url = task_info.get('repo_url')
        base_commit = task_info.get('base_commit', 'HEAD')

        logger.info(f"Processing SWE-bench task: {task_id}")

        # Clone repository
        cloned_path = None
        try:
            if repo_url and task_id:
                logger.info(f"Cloning {repo_url} for task {task_id}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Cloning repository...", updater.context_id, updater.task_id)
                )
                cloned_path = clone_repository(repo_url, base_commit, task_id)
                set_repo_path(cloned_path)
                logger.info(f"Repository cloned to: {cloned_path}")
            else:
                # Use default path
                set_repo_path(os.getenv("REPO_PATH", "/repo"))

            # Create fresh agent for this task
            logger.info("Creating fresh agent instance")
            agent = self._create_fresh_agent()

            # Stream the agent response
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Analyzing issue and generating patch...", updater.context_id, updater.task_id)
            )

            response_text = ""
            async for event in agent.stream_async(user_input):
                if "data" in event:
                    if text_content := event["data"]:
                        response_text += text_content
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(text_content, updater.context_id, updater.task_id)
                        )
                elif "result" in event:
                    result = event["result"]
                    if result:
                        response_text = str(result)

            # Extract and format patch
            final_response = extract_patch(response_text)

            # Send final response
            await updater.add_artifact(
                [Part(root=TextPart(text=final_response))],
                name="patch_response"
            )
            await updater.complete()

            logger.info(f"Task {task_id} completed successfully")

            # Cleanup agent reference
            del agent

        except Exception as e:
            logger.exception(f"Error executing task {task_id}: {e}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {str(e)}", updater.context_id, updater.task_id)
            )

        finally:
            # ALWAYS cleanup the cloned repository
            if cloned_path:
                logger.info(f"Cleaning up repository: {cloned_path}")
                cleanup_repository(cloned_path)

            # Force garbage collection to free memory
            gc.collect()

            # Periodically cleanup all old repos to prevent disk exhaustion
            if task_num % self.CLEANUP_INTERVAL == 0:
                self._cleanup_all_repos()

            self._log_memory_usage(f"Task #{task_num} end")
            logger.info(f"Task #{task_num} cleanup complete")

    def _cleanup_all_repos(self) -> None:
        """Remove all cloned repositories to free disk space."""
        try:
            if Path(REPOS_DIR).exists():
                repo_count = len(list(Path(REPOS_DIR).iterdir()))
                if repo_count > 0:
                    logger.info(f"Periodic cleanup: removing {repo_count} repos from {REPOS_DIR}")
                    shutil.rmtree(REPOS_DIR, ignore_errors=True)
                    os.makedirs(REPOS_DIR, exist_ok=True)
        except Exception as e:
            logger.warning(f"Error in periodic cleanup: {e}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported."""
        from a2a.types import UnsupportedOperationError
        raise ServerError(error=UnsupportedOperationError())


def create_agent(
    api_key: Optional[str] = None,
    model_id: str = "gpt-5-mini-2025-08-07",
    max_tokens: int = 4096
) -> Agent:
    """Create a Strands Agent configured for SWE-bench tasks."""

    # Get API key from parameter or environment
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")

    # Configure OpenAI model
    model = OpenAIModel(
        client_args={"api_key": api_key},
        model_id=model_id,
        params={
            "max_completion_tokens": max_tokens
        }
    )

    # Create the agent with repository exploration tools
    # PrintingCallbackHandler shows tool calls and responses in console
    agent = Agent(
        name="SWE-bench White Agent",
        description="A coding agent that analyzes GitHub issues and generates unified diff patches to fix bugs.",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[read_file, list_directory, search_code, find_files, get_file_info],
        callback_handler=PrintingCallbackHandler()
    )

    return agent


def create_a2a_server(
    agent: Agent,
    host: str = "0.0.0.0",
    port: int = 9002,
    public_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: str = "gpt-5-mini-2025-08-07",
    max_tokens: int = 4096
) -> A2AServer:
    """Create an A2A server with custom executor for SWE-bench tasks.

    Uses a custom executor that creates a fresh agent for each task,
    ensuring no conversation history is shared between tasks and
    repositories are always cleaned up.
    """
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore

    # Define skills for the agent card
    skills = [
        {
            "id": "solve_swebench",
            "name": "SWE-bench Solver",
            "description": "Receives a GitHub issue and generates a unified diff patch to fix it.",
            "tags": ["white agent", "coding", "swe-bench", "patch generation"]
        }
    ]

    # Build server kwargs
    server_kwargs = {
        "agent": agent,
        "host": host,
        "port": port,
        "version": "1.0.0",
        "skills": skills
    }

    # Add public URL if provided (for agent card)
    if public_url:
        server_kwargs["http_url"] = public_url
        # Controller proxies to root, so serve at root even with path in URL
        server_kwargs["serve_at_root"] = True

    server = A2AServer(**server_kwargs)

    # Replace the default executor with our custom one that:
    # 1. Creates a fresh agent per task (no shared conversation history)
    # 2. Clones and cleans up repositories for each task
    # 3. Properly manages memory between tasks
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not effective_api_key:
        raise ValueError("OPENAI_API_KEY required for custom executor")

    custom_executor = SWEBenchA2AExecutor(
        api_key=effective_api_key,
        model_id=model_id,
        max_tokens=max_tokens
    )

    # Replace the request handler with one using our custom executor
    server.request_handler = DefaultRequestHandler(
        agent_executor=custom_executor,
        task_store=InMemoryTaskStore()
    )

    logger.info("A2A server configured with SWEBenchA2AExecutor (fresh agent per task)")

    return server


class SWEBenchWhiteAgentExecutor:
    """Executor class for SWE-bench white agent tasks."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gpt-5-mini-2025-08-07",
        max_tokens: int = 4096
    ):
        self.agent = create_agent(
            api_key=api_key,
            model_id=model_id,
            max_tokens=max_tokens
        )

    def execute(
        self,
        user_input: str,
        repo_path: Optional[str] = None,
        cleanup: bool = False
    ) -> str:
        """Execute a SWE-bench task and return the patch response.

        Args:
            user_input: The task description with XML tags
            repo_path: Optional path to pre-cloned repository (skips cloning)
            cleanup: Whether to remove the cloned repo after execution
        """
        # Parse the input to extract all fields
        task_info = parse_task_input(user_input)

        task_id = task_info.get('task_id', 'Unknown')
        repo = task_info.get('repo', 'Unknown')
        repo_url = task_info.get('repo_url')
        base_commit = task_info.get('base_commit')

        print(f"Processing task: {task_id}")
        print(f"Repository: {repo}")

        # Determine the repository path
        # Priority: explicit param > pre-cloned path > clone from URL > env var
        effective_repo_path = None
        cloned_path = None

        if repo_path:
            # Use explicitly provided path
            effective_repo_path = repo_path
            print(f"Using provided repo path: {effective_repo_path}")

        elif task_info.get('repo_path'):
            # Use path from XML tag
            effective_repo_path = task_info['repo_path']
            print(f"Using repo path from task: {effective_repo_path}")

        elif repo_url and base_commit and task_id:
            # Clone the repository
            print(f"Cloning from: {repo_url}")
            print(f"Base commit: {base_commit}")
            try:
                cloned_path = clone_repository(repo_url, base_commit, task_id)
                effective_repo_path = cloned_path
            except Exception as e:
                print(f"Error cloning repository: {e}")
                # Continue without repo - agent will do its best
                effective_repo_path = os.getenv("REPO_PATH", "/repo")

        else:
            # Fallback to environment variable or default
            effective_repo_path = os.getenv("REPO_PATH", "/repo")
            print(f"Using default repo path: {effective_repo_path}")

        # Set the repository path for tools
        set_repo_path(effective_repo_path)
        print(f"Repository path set to: {effective_repo_path}")

        try:
            # Pass the raw input directly - the system prompt explains the XML format
            # The agent will use tools to explore the repo and generate a patch
            response = self.agent(user_input)

            # Extract response text from Strands response object
            if hasattr(response, 'message') and response.message:
                msg = response.message
                # Handle message with content list (OpenAI format)
                if isinstance(msg, dict) and 'content' in msg:
                    content = msg['content']
                    if isinstance(content, list):
                        # Extract text from content blocks
                        texts = [c.get('text', '') for c in content if isinstance(c, dict) and 'text' in c]
                        response_text = '\n'.join(texts)
                    else:
                        response_text = str(content)
                else:
                    response_text = str(msg)
            elif hasattr(response, 'text'):
                response_text = str(response.text)
            else:
                response_text = str(response)

            # Ensure patch is properly wrapped in <patch> tags
            return extract_patch(response_text)

        finally:
            # Cleanup cloned repository if requested
            if cleanup and cloned_path:
                print(f"Cleaning up cloned repository: {cloned_path}")
                cleanup_repository(cloned_path)

    async def execute_async(
        self,
        user_input: str,
        repo_path: Optional[str] = None,
        cleanup: bool = False
    ) -> str:
        """Async version of execute."""
        # For now, just call sync version
        # Strands supports async but keeping it simple
        return self.execute(user_input, repo_path=repo_path, cleanup=cleanup)

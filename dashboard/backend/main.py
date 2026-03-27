"""
Auto App Generation — Orchestration Server

FastAPI + Socket.IO backend that controls Gemini CLI and Claude Code CLI
as subprocesses and streams their output to the dashboard in real-time.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agents import ClaudeAgent, StitchAgent
from agents.planner_agent import PlannerAgent
from models import (
    Artifact,
    LogEntry,
    LogPrefix,
    PipelineState,
    PipelineStep,
    StartRequest,
    StepStatus,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATED_APP_DIR = PROJECT_ROOT / "generated-app"
DOCS_DIR = PROJECT_ROOT / "docs"

# Load .env from project root
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Auto App Generation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

pipeline_state = PipelineState()
artifacts: list[dict] = []
planner_agent: Optional[PlannerAgent] = None
claude_agent: Optional[ClaudeAgent] = None
current_task: Optional[asyncio.Task] = None
current_project_dir: Optional[Path] = None  # Tracks current generated app directory
pipeline_start_time: Optional[float] = None  # Track pipeline elapsed time
step_start_times: dict[str, float] = {}  # Track per-step elapsed time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def emit_log(agent_name: str, prefix: str, content: str):
    """Send a log entry to all connected clients."""
    entry = LogEntry(
        prefix=LogPrefix(prefix),
        content=content,
        agent=agent_name,
    )
    await sio.emit("log", entry.model_dump())


async def emit_pipeline_state():
    """Broadcast current pipeline state with elapsed time."""
    data = pipeline_state.model_dump()
    if pipeline_start_time:
        data["elapsed_seconds"] = round(time.time() - pipeline_start_time)
    data["step_times"] = {k: round(v, 1) for k, v in step_start_times.items()}
    await sio.emit("pipeline_state", data)


async def emit_agent_status(agent: str, status: str):
    """Broadcast agent status change."""
    await sio.emit("agent_status", {"agent": agent, "status": status})


async def emit_artifact(artifact: dict):
    """Broadcast new artifact."""
    artifacts.append(artifact)
    await sio.emit("artifact", artifact)


async def emit_file_tree():
    """Scan current project directory and emit file tree."""
    scan_path = current_project_dir if current_project_dir else GENERATED_APP_DIR
    tree = _scan_dir(scan_path)
    await sio.emit("file_tree", tree)


def _scan_dir(path: Path, depth: int = 0) -> list[dict]:
    """Recursively scan directory into a tree structure."""
    if not path.exists() or depth > 5:
        return []

    items = []
    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    except PermissionError:
        return []

    skip = {"node_modules", ".next", "__pycache__", ".git", "venv", ".venv"}

    for entry in entries:
        if entry.name.startswith(".") and entry.name not in (".env.example",):
            continue
        if entry.name in skip:
            continue

        node = {
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "is_new": True,
            "children": [],
        }

        if entry.is_dir():
            node["children"] = _scan_dir(entry, depth + 1)

        items.append(node)

    return items


def _make_sonnet_log_callback(agent_name: str = "sonnet"):
    async def callback(prefix: str, content: str):
        await emit_log(agent_name, prefix, content)
    return callback


def _make_claude_log_callback(agent_name: str = "claude"):
    async def callback(prefix: str, content: str):
        await emit_log(agent_name, prefix, content)
    return callback


# ---------------------------------------------------------------------------
# MoE Gating — AI-based complexity analysis to select & configure agents
# ---------------------------------------------------------------------------


async def _moe_gate_implementation_ai(claude_md: str, planner_agent) -> dict:
    """MoE Gating Phase: Use 3 Gemini instances in parallel to analyze
    project complexity across FE/BE/UI dimensions.

    Returns: {
        "fe": {"score": 8, "notes": "...", "active": True, "mode": "senior"},
        "be": {"score": 6, "notes": "...", "active": True, "mode": "normal"},
        "ui": {"score": 3, "notes": "...", "active": False, "mode": "skip"},
    }
    """
    import re as _re

    gating_prompt_template = """아래 CLAUDE.md를 분석해서 {domain} 복잡도를 평가해줘.
JSON으로만 출력하세요. 다른 텍스트 없이:

```json
{{
  "score": 7,
  "reasons": ["이유1", "이유2", "이유3"],
  "key_tasks": ["주요작업1", "주요작업2"]
}}
```

score는 1~10 (1=거의 없음, 10=매우 복잡)

CLAUDE.md:
{claude_md}"""

    domains = {
        "fe": "프론트엔드 (페이지 수, 컴포넌트 복잡도, 상태관리, 인터랙션)",
        "be": "백엔드 (API 수, DB 테이블, 인증, 비즈니스 로직)",
        "ui": "디자인/UI (커스텀 디자인 요구, 애니메이션, 테마, 반응형)",
    }

    async def analyze_domain(key: str, domain_desc: str) -> tuple[str, dict]:
        prompt = gating_prompt_template.format(domain=domain_desc, claude_md=claude_md[:3000])
        agent = PlannerAgent(planner_agent.work_dir, planner_agent.on_log)
        result = await agent.run_prompt(prompt, _internal=True)

        # Parse JSON
        try:
            json_match = _re.search(r'\{[\s\S]*\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                return key, parsed
        except Exception:
            pass
        return key, {"score": 5, "reasons": ["파싱 실패 — 기본값"], "key_tasks": []}

    # Run Gating Agents sequentially with delay (avoid 429 rate limit)
    await emit_log("sonnet", "SYS", "MoE Gating Phase 시작 (3개 분석 에이전트 순차 실행)")

    results = []
    for k, v in domains.items():
        try:
            result = await analyze_domain(k, v)
            results.append(result)
        except Exception as e:
            results.append(e)
        await asyncio.sleep(2)  # Rate limit delay

    gating: dict[str, dict] = {}
    for r in results:
        if isinstance(r, Exception):
            continue
        key, data = r
        score = data.get("score", 5)
        reasons = data.get("reasons", [])
        key_tasks = data.get("key_tasks", [])

        if score >= 7:
            mode = "senior"
        elif score >= 4:
            mode = "normal"
        else:
            mode = "skip"

        gating[key] = {
            "score": score,
            "reasons": reasons,
            "key_tasks": key_tasks,
            "active": score >= 4,
            "mode": mode,
        }

        emoji = "🔴" if score >= 7 else "🟡" if score >= 4 else "🟢"
        await emit_log("sonnet", "SNT",
            f"[GATE-{key.upper()}] {emoji} 복잡도 {score}/10 → {mode} | {', '.join(reasons[:2])}")

    # Ensure at least FE is active
    if not any(g.get("active") for g in gating.values()):
        gating.setdefault("fe", {})["active"] = True
        gating["fe"]["mode"] = "normal"
        gating["fe"]["score"] = 5

    return gating


def _moe_gate_review(code_summary: str) -> list[dict]:
    """Analyze code to determine which review experts to activate.

    Returns a subset of 5 experts based on what's in the code.
    """
    summary_lower = code_summary.lower()

    all_experts = [
        {
            "name": "아키텍처 전문가",
            "focus": "시스템 설계, 컴포넌트 분리, 폴더 구조, 확장성, 의존성 관리",
            "prefix": "ARCH",
            "triggers": ["import", "router", "middleware", "module", "service", "controller"],
            "always": True,  # Always include architecture review
        },
        {
            "name": "보안 전문가",
            "focus": "인증/인가, 입력 검증, XSS/SQL Injection, 시크릿 노출, OWASP Top 10",
            "prefix": "SEC",
            "triggers": ["auth", "login", "password", "token", "jwt", "session", "cookie",
                        "api_key", "secret", ".env", "sql", "query", "input"],
            "always": False,
        },
        {
            "name": "성능 전문가",
            "focus": "불필요한 리렌더링, N+1 쿼리, 번들 크기, 캐싱, 비동기 처리",
            "prefix": "PERF",
            "triggers": ["useEffect", "useState", "query", "fetch", "async", "await",
                        "database", "cache", "index", "loop", "map", "filter"],
            "always": False,
        },
        {
            "name": "UX/디자인 전문가",
            "focus": "반응형 디자인, 접근성(a11y), 사용자 흐름, 로딩 상태, 에러 상태 처리",
            "prefix": "UX",
            "triggers": ["css", "style", "className", "responsive", "mobile", "button",
                        "input", "form", "modal", "loading", "error", "aria"],
            "always": False,
        },
        {
            "name": "코드 품질 전문가",
            "focus": "네이밍, 중복 코드, 타입 안정성, 에러 처리, 가독성, 미사용 import",
            "prefix": "QUAL",
            "triggers": ["any", "TODO", "FIXME", "console.log", "print(", "except:", "catch"],
            "always": True,  # Always include quality review
        },
    ]

    activated = []
    for expert in all_experts:
        if expert["always"]:
            activated.append(expert)
            continue
        # Check if any trigger keyword exists in the code
        if any(trigger in summary_lower for trigger in expert["triggers"]):
            activated.append(expert)

    # Minimum 3 experts
    if len(activated) < 3:
        for expert in all_experts:
            if expert not in activated:
                activated.append(expert)
                if len(activated) >= 3:
                    break

    return activated


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Create a filesystem-safe slug from text."""
    import re
    # Keep alphanumeric, Korean chars, hyphens, underscores
    slug = re.sub(r'[^\w\s가-힣-]', '', text.strip().lower())
    slug = re.sub(r'[\s_]+', '-', slug)
    return slug[:50] or "untitled"


async def run_pipeline(prompt: str, project_name: Optional[str] = None, google_api_key: str = ""):
    """Execute the full 5-stage pipeline."""
    global planner_agent, claude_agent, pipeline_state, current_project_dir
    global pipeline_start_time, step_start_times

    pipeline_start_time = time.time()
    step_start_times = {}

    # Create project-specific directory under generated-app/
    slug = _slugify(project_name or prompt)
    project_dir = GENERATED_APP_DIR / slug
    # Avoid collision
    if project_dir.exists():
        i = 2
        while (GENERATED_APP_DIR / f"{slug}-{i}").exists():
            i += 1
        project_dir = GENERATED_APP_DIR / f"{slug}-{i}"

    project_dir.mkdir(parents=True, exist_ok=True)
    current_project_dir = project_dir
    work_dir = str(project_dir)

    await emit_log("sonnet", "SYS", f"프로젝트 폴더: generated-app/{project_dir.name}/")

    async def on_gemini_question():
        """Called when Gemini is waiting for user input (legacy)."""
        await emit_agent_status("sonnet", "waiting")
        await sio.emit("waiting_for_input", {"agent": "sonnet"})

    async def on_gemini_question_structured(question_data: dict):
        """Send structured question with options to frontend."""
        await emit_agent_status("sonnet", "waiting")
        await sio.emit("structured_question", {
            "agent": "sonnet",
            "id": question_data.get("id", ""),
            "text": question_data.get("text", ""),
            "options": question_data.get("options", []),
            "multi_select": question_data.get("multi_select", False),
        })

    planner_agent = PlannerAgent(
        work_dir, _make_sonnet_log_callback(),
        on_question=on_gemini_question,
        on_question_structured=on_gemini_question_structured,
    )
    claude_agent = ClaudeAgent(work_dir, _make_claude_log_callback())

    try:
        # ── Stage 1: Envisioning (Interactive Q&A) ──
        step_start_times["envisioning_start"] = time.time()
        pipeline_state.advance_to(PipelineStep.ENVISIONING)
        await emit_pipeline_state()
        await emit_agent_status("sonnet", "running")

        spec = await planner_agent.run_envisioning_interactive(prompt)

        step_start_times["envisioning"] = time.time() - step_start_times.pop("envisioning_start")
        await emit_agent_status("sonnet", "idle")
        await emit_artifact({
            "title": "spec.md — 프로젝트 기획서",
            "description": "Gemini가 생성한 요구사항 명세서",
            "file_path": "docs/01-planning/spec.md",
            "size": f"{len(spec.encode('utf-8')) / 1024:.1f} KB",
            "created_at": datetime.now().strftime("%H:%M"),
            "created_by": "sonnet",
            "icon_type": "md",
        })

        if not spec.strip():
            await emit_log("sonnet", "ERR", "기획서 생성 실패 — 파이프라인 중단")
            return

        # ── Stage 2: Blueprinting ──
        step_start_times["blueprinting_start"] = time.time()
        pipeline_state.advance_to(PipelineStep.BLUEPRINTING)
        await emit_pipeline_state()
        await emit_agent_status("sonnet", "running")

        claude_md = await planner_agent.run_blueprinting(spec)

        step_start_times["blueprinting"] = time.time() - step_start_times.pop("blueprinting_start")
        await emit_agent_status("sonnet", "idle")
        await emit_artifact({
            "title": "CLAUDE.md — 작업 지침서",
            "description": "Claude Code가 참조할 구현 가이드",
            "file_path": "CLAUDE.md",
            "size": f"{len(claude_md.encode('utf-8')) / 1024:.1f} KB",
            "created_at": datetime.now().strftime("%H:%M"),
            "created_by": "sonnet",
            "icon_type": "md",
        })

        if not claude_md.strip():
            await emit_log("sonnet", "ERR", "CLAUDE.md 생성 실패 — 파이프라인 중단")
            return

        # ── Stage 3: Implementation ──
        # Inject API key into .env if provided
        if google_api_key:
            env_path = project_dir / ".env"
            with open(env_path, "w") as f:
                f.write(f"GOOGLE_API_KEY={google_api_key}\n")
            await emit_log("claude", "SYS", ".env 파일에 GOOGLE_API_KEY 주입됨")

        step_start_times["implementation_start"] = time.time()
        pipeline_state.advance_to(PipelineStep.IMPLEMENTATION)
        await emit_pipeline_state()
        await emit_agent_status("claude", "running")

        # Periodically refresh file tree while Claude is working
        async def poll_file_tree():
            try:
                while True:
                    await emit_file_tree()
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass

        poll_task = asyncio.create_task(poll_file_tree())

        # Tagged log callback for parallel mode
        async def on_log_tagged(tag: str, prefix: str, content: str):
            await emit_log("claude", prefix, f"[{tag}] {content}")

        # ── MoE Gating Phase: AI analyzes complexity ──
        gating = await _moe_gate_implementation_ai(claude_md, planner_agent)

        active_agents = [k for k, v in gating.items() if v.get("active")]
        if not active_agents:
            active_agents = ["fe"]

        # ── Stage 3a: Stitch 2.0 UI Design (replaces Claude UI Agent) ──
        stitch_api_key = os.environ.get("STITCH_API_KEY", "")
        if "ui" in active_agents and stitch_api_key:
            active_agents = [a for a in active_agents if a != "ui"]  # Remove Claude UI agent
            await emit_log("sonnet", "SYS", "Stitch 2.0으로 UI 디자인 생성 중...")

            stitch = StitchAgent(work_dir, _make_sonnet_log_callback(), api_key=stitch_api_key)

            # Extract screen names from CLAUDE.md
            screen_names = []
            for line in claude_md.split("\n"):
                if "page" in line.lower() or "페이지" in line or "화면" in line:
                    clean = line.strip().lstrip("-* ").strip()
                    if clean and len(clean) < 60:
                        screen_names.append(clean)
            if not screen_names:
                screen_names = ["메인 홈", "대시보드", "설정"]

            design_result = await stitch.generate_screens(
                spec[:500] if spec else "앱 UI 디자인",
                screen_names[:5],
            )

            await emit_artifact({
                "title": "Stitch UI 디자인",
                "description": "Stitch 2.0이 생성한 UI 디자인",
                "file_path": "stitch-design-guide.json",
                "size": f"{len(json.dumps(design_result).encode('utf-8')) / 1024:.1f} KB",
                "created_at": datetime.now().strftime("%H:%M"),
                "created_by": "sonnet",
                "icon_type": "review",
            })

        scores_str = ", ".join(f"{k.upper()}:{v.get('score',0)}점" for k, v in gating.items())
        await emit_log("claude", "SYS",
            f"MoE Gating 완료 → 활성 Agent: {[a.upper() for a in active_agents]} ({scores_str})")

        # Inject gating analysis into agent prompts
        gating_context = gating

        # Parallel implementation with isolated workspaces + gating context
        impl_result = await claude_agent.run_implementation_parallel(
            str(project_dir / "CLAUDE.md"), on_log_tagged,
            active_agents=active_agents, gating_context=gating_context
        )

        poll_task.cancel()
        step_start_times["implementation"] = time.time() - step_start_times.pop("implementation_start")
        await emit_agent_status("claude", "idle")
        await emit_file_tree()

        # ── Stage 4: Review ──
        pipeline_state.advance_to(PipelineStep.REVIEW)
        await emit_pipeline_state()
        await emit_agent_status("sonnet", "running")

        # Gather code summary for review
        code_files = []
        for ext in ("*.ts", "*.tsx", "*.py", "*.js", "*.jsx"):
            code_files.extend(project_dir.rglob(ext))

        code_summary_parts = []
        for cf in code_files[:15]:  # Limit to 15 files
            try:
                content = cf.read_text(encoding="utf-8")
                rel = cf.relative_to(project_dir)
                code_summary_parts.append(f"### {rel}\n```\n{content[:2000]}\n```")
            except Exception:
                pass

        code_summary = "\n\n".join(code_summary_parts) if code_summary_parts else impl_result

        # MoE Gating for review: select experts based on code content
        active_experts = _moe_gate_review(code_summary)
        await emit_log("sonnet", "SYS",
            f"MoE Gating → 활성 전문가: {[e['prefix'] for e in active_experts]} ({len(active_experts)}명)")

        step_start_times["review_start"] = time.time()
        review = await planner_agent.run_review_moe(code_summary, experts=active_experts)
        step_start_times["review"] = time.time() - step_start_times.pop("review_start")

        await emit_agent_status("sonnet", "idle")
        await emit_artifact({
            "title": "review.md — 코드 리뷰 리포트",
            "description": "Gemini의 코드 리뷰 결과",
            "file_path": "docs/04-reviews/review.md",
            "size": f"{len(review.encode('utf-8')) / 1024:.1f} KB",
            "created_at": datetime.now().strftime("%H:%M"),
            "created_by": "sonnet",
            "icon_type": "review",
        })

        # ── Stage 5: Feedback loop (1 round) ──
        import re
        has_critical = bool(re.search(r'🔴|Critical.*수정|즉시 수정', review, re.IGNORECASE))
        if review.strip() and has_critical:
            pipeline_state.advance_to(PipelineStep.FEEDBACK)
            await emit_pipeline_state()
            await emit_agent_status("claude", "running")

            await claude_agent.run_feedback(review)

            await emit_agent_status("claude", "idle")
            await emit_file_tree()

        # ── Done ──
        for step in pipeline_state.steps:
            pipeline_state.steps[step] = StepStatus.DONE
        await emit_pipeline_state()

        total_time = round(time.time() - pipeline_start_time) if pipeline_start_time else 0
        await emit_log("sonnet", "SYS", f"파이프라인 완료! (총 {total_time // 60}분 {total_time % 60}초)")

        # ── Auto-launch generated app on port 3001 ──
        await _auto_launch_app(project_dir)

    except asyncio.CancelledError:
        await emit_log("sonnet", "SYS", "파이프라인이 취소되었습니다.")
    except Exception as e:
        await emit_log("sonnet", "ERR", f"파이프라인 오류: {str(e)}")


app_process: Optional[asyncio.subprocess.Process] = None


async def _auto_launch_app(project_dir: Path):
    """Detect app type and launch on port 3001."""
    global app_process

    # Kill any existing app on 3001
    if app_process and app_process.returncode is None:
        app_process.terminate()
        await asyncio.sleep(1)

    try:
        kill_proc = await asyncio.create_subprocess_shell(
            "lsof -ti:3001 | xargs kill 2>/dev/null; true",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await kill_proc.wait()
        await asyncio.sleep(1)
    except Exception:
        pass

    await emit_log("sonnet", "SYS", "생성된 앱을 3001번 포트에 실행합니다...")

    # Detect app type
    is_flutter = (project_dir / "pubspec.yaml").exists()
    is_nextjs = (project_dir / "package.json").exists() and (project_dir / "next.config.ts").exists() or (project_dir / "next.config.js").exists()
    has_frontend_dir = (project_dir / "frontend" / "package.json").exists()
    has_flutter_frontend = (project_dir / "frontend" / "pubspec.yaml").exists()
    has_vite = (project_dir / "package.json").exists() and (project_dir / "vite.config.ts").exists()
    has_frontend_vite = has_frontend_dir and (project_dir / "frontend" / "vite.config.ts").exists()

    try:
        if is_flutter:
            # Flutter Web — build and serve
            web_dir = project_dir / "build" / "web"
            if not web_dir.exists():
                await emit_log("sonnet", "SYS", "Flutter Web 빌드 중...")
                build = await asyncio.create_subprocess_exec(
                    "flutter", "build", "web",
                    cwd=str(project_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await build.wait()
            if web_dir.exists():
                app_process = await asyncio.create_subprocess_exec(
                    "python3", "-m", "http.server", "3001",
                    cwd=str(web_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                await emit_log("sonnet", "ERR", "Flutter Web 빌드 실패")
                return

        elif has_flutter_frontend:
            web_dir = project_dir / "frontend" / "build" / "web"
            if web_dir.exists():
                app_process = await asyncio.create_subprocess_exec(
                    "python3", "-m", "http.server", "3001",
                    cwd=str(web_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                await emit_log("sonnet", "ERR", "Flutter 빌드 폴더가 없습니다")
                return

        elif has_frontend_vite or has_vite:
            # Vite project — try dist, else dev
            app_dir = project_dir / "frontend" if has_frontend_vite else project_dir
            dist_dir = app_dir / "dist"
            if dist_dir.exists():
                app_process = await asyncio.create_subprocess_exec(
                    "python3", "-m", "http.server", "3001",
                    cwd=str(dist_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                app_process = await asyncio.create_subprocess_exec(
                    "npx", "vite", "--port", "3001", "--host",
                    cwd=str(app_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

        elif is_nextjs or has_frontend_dir:
            # Next.js — build and start
            app_dir = project_dir / "frontend" if has_frontend_dir and not is_nextjs else project_dir
            # Install deps if needed
            if not (app_dir / "node_modules").exists():
                await emit_log("sonnet", "SYS", "npm install 중...")
                install = await asyncio.create_subprocess_exec(
                    "npm", "install",
                    cwd=str(app_dir),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await install.wait()
            # Build
            await emit_log("sonnet", "SYS", "Next.js 빌드 중...")
            build = await asyncio.create_subprocess_exec(
                "npx", "next", "build",
                cwd=str(app_dir),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await build.wait()
            # Start
            app_process = await asyncio.create_subprocess_exec(
                "npx", "next", "start", "--port", "3001",
                cwd=str(app_dir),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        else:
            # Fallback — try to find any index.html
            for candidate in [project_dir, project_dir / "dist", project_dir / "build", project_dir / "public"]:
                if (candidate / "index.html").exists():
                    app_process = await asyncio.create_subprocess_exec(
                        "python3", "-m", "http.server", "3001",
                        cwd=str(candidate),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    break
            else:
                await emit_log("sonnet", "SYS", "자동 실행 가능한 앱 형태를 감지하지 못했습니다")
                return

        await asyncio.sleep(3)
        await emit_log("sonnet", "SYS", "앱이 http://localhost:3001 에서 실행 중입니다!")
        await sio.emit("app_launched", {"url": "http://localhost:3001"})

    except Exception as e:
        await emit_log("sonnet", "ERR", f"앱 실행 오류: {str(e)}")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return {"status": "ok", "sonnet": planner_agent is not None and planner_agent.is_running, "claude": claude_agent is not None and claude_agent.is_running}


@app.get("/api/pipeline")
async def get_pipeline():
    return pipeline_state.model_dump()


@app.get("/api/artifacts")
async def get_artifacts():
    return artifacts


@app.get("/api/files")
async def get_files():
    scan_path = current_project_dir if current_project_dir else GENERATED_APP_DIR
    return _scan_dir(scan_path)


@app.get("/api/projects")
async def get_projects():
    """List all generated projects."""
    projects = []
    if GENERATED_APP_DIR.exists():
        for entry in sorted(GENERATED_APP_DIR.iterdir(), key=lambda e: e.stat().st_mtime, reverse=True):
            if entry.is_dir() and not entry.name.startswith("."):
                has_claude_md = (entry / "CLAUDE.md").exists()
                has_package = (entry / "package.json").exists() or (entry / "pubspec.yaml").exists()
                file_count = sum(1 for _ in entry.rglob("*") if _.is_file() and "node_modules" not in str(_) and ".next" not in str(_))
                projects.append({
                    "name": entry.name,
                    "path": str(entry),
                    "has_claude_md": has_claude_md,
                    "has_package": has_package,
                    "file_count": file_count,
                    "created_at": datetime.fromtimestamp(entry.stat().st_ctime).strftime("%Y-%m-%d %H:%M"),
                })
    return projects


@app.post("/api/stop")
async def stop_pipeline():
    global current_task
    if current_task and not current_task.done():
        current_task.cancel()
    if planner_agent:
        await planner_agent.stop()
    if claude_agent:
        await claude_agent.stop()
    return {"status": "stopped"}


@app.post("/api/launch")
async def launch_project(request: StartRequest):
    """Launch a previous project on port 3001."""
    project_path = Path(request.prompt) if request.prompt else None
    if not project_path or not project_path.exists():
        return {"status": "error", "message": "Project path not found"}

    global current_project_dir
    current_project_dir = project_path
    await _auto_launch_app(project_path)
    return {"status": "ok", "url": "http://localhost:3001"}


@app.post("/api/deploy")
async def deploy_app():
    """Deploy the current generated app to Vercel."""
    if not current_project_dir or not current_project_dir.exists():
        return {"status": "error", "message": "No project to deploy"}

    try:
        # Find the deployable directory
        deploy_dir = current_project_dir
        if (current_project_dir / "frontend").exists():
            deploy_dir = current_project_dir / "frontend"

        proc = await asyncio.create_subprocess_exec(
            "vercel", "--yes", "--prod",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(deploy_dir),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode("utf-8", errors="replace").strip()

        # Extract URL from vercel output
        url = ""
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("https://"):
                url = line
                break

        if url:
            await sio.emit("deployed", {"url": url})
            return {"status": "ok", "url": url}
        else:
            return {"status": "error", "message": output[:200]}

    except asyncio.TimeoutError:
        return {"status": "error", "message": "Deploy timeout (120s)"}
    except FileNotFoundError:
        return {"status": "error", "message": "vercel CLI not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------


@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    # If pipeline is not actively running, send clean initial state
    is_active = any(v == "active" for v in pipeline_state.steps.values())
    if is_active:
        await sio.emit("pipeline_state", pipeline_state.model_dump(), to=sid)
    else:
        await sio.emit("pipeline_state", PipelineState().model_dump(), to=sid)
    await sio.emit("artifacts_list", artifacts, to=sid)
    scan_path = current_project_dir if current_project_dir else GENERATED_APP_DIR
    tree = _scan_dir(scan_path)
    await sio.emit("file_tree", tree, to=sid)


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


@sio.event
async def start_pipeline(sid, data):
    """Start the full pipeline from user input."""
    global current_task, pipeline_state, artifacts

    # Reset state
    pipeline_state = PipelineState()
    artifacts.clear()

    prompt = data.get("prompt", "")
    project_name = data.get("project_name")
    google_api_key = data.get("google_api_key", "")

    if not prompt:
        await sio.emit("log", {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "prefix": "ERR",
            "content": "프롬프트가 비어 있습니다.",
            "agent": "sonnet",
        }, to=sid)
        return

    current_task = asyncio.create_task(run_pipeline(prompt, project_name, google_api_key))


@sio.event
async def send_to_agent(sid, data):
    """Send a message to a specific agent."""
    agent_name = data.get("agent", "")
    message = data.get("message", "")

    if not message:
        return

    if agent_name == "sonnet" and planner_agent:
        # Check if Gemini is waiting for a Q&A answer
        if planner_agent._user_response and not planner_agent._user_response.done():
            # Don't log USR here — run_envisioning_interactive will log it
            await planner_agent.send_user_response(message)
            await emit_agent_status("sonnet", "running")
        else:
            await emit_log(agent_name, "USR", message)
            await emit_agent_status("sonnet", "running")
            result = await planner_agent.run_prompt(message)
            await emit_agent_status("sonnet", "idle")
    elif agent_name == "claude" and claude_agent:
        await emit_log(agent_name, "USR", message)
        await emit_agent_status("claude", "running")
        result = await claude_agent.run_custom(message)
        await emit_agent_status("claude", "idle")
        await emit_file_tree()
    else:
        await emit_log(agent_name, "ERR", "에이전트가 초기화되지 않았습니다. 먼저 파이프라인을 시작해주세요.")


@sio.event
async def stop(sid, data=None):
    """Stop all running agents."""
    await stop_pipeline()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

# Export the ASGI app (socket_app wraps FastAPI)
app_asgi = socket_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app_asgi",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )

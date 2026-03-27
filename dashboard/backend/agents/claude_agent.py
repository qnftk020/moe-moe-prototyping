"""Claude Code CLI agent - manages subprocess interaction with claude CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Callable, Optional

CLAUDE_BIN = shutil.which("claude") or "claude"


class ClaudeAgent:
    """Controls Claude Code CLI as a subprocess for implementation tasks."""

    def __init__(self, work_dir: str, on_log: Callable):
        self.work_dir = work_dir
        self.on_log = on_log  # async callback(prefix, content)
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_running = False

    async def run_prompt(self, prompt: str) -> str:
        """Run claude in non-interactive print mode with streaming JSON output."""
        self.is_running = True
        await self.on_log("SYS", f"Claude Code 세션 시작됨 (프로젝트: {os.path.basename(self.work_dir)}/)")

        cmd = [
            CLAUDE_BIN,
            "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
                env={**os.environ, "NO_COLOR": "1"},
            )

            full_result: list[str] = []
            buffer = ""

            async def read_stdout():
                nonlocal buffer
                while True:
                    chunk = await self.process.stdout.read(4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    buffer += text

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        await self._process_stream_line(line, full_result)

                if buffer.strip():
                    await self._process_stream_line(buffer.strip(), full_result)
                    buffer = ""

            async def read_stderr():
                while True:
                    chunk = await self.process.stderr.read(256)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace").strip()
                    if text and not _is_ignorable_stderr(text):
                        await self.on_log("ERR", text)

            await asyncio.gather(read_stdout(), read_stderr())
            await self.process.wait()

            return "\n".join(full_result)

        except FileNotFoundError:
            await self.on_log("ERR", f"claude CLI를 찾을 수 없습니다. 경로: {CLAUDE_BIN}")
            return ""
        except Exception as e:
            await self.on_log("ERR", f"Claude 실행 오류: {str(e)}")
            return ""
        finally:
            self.is_running = False
            self.process = None

    async def _process_stream_line(self, line: str, result_collector: list[str]):
        """Process a single line from Claude's stream-json output."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            if line.strip():
                await self.on_log("OPS", line)
                result_collector.append(line)
            return

        msg_type = data.get("type", "")

        # Skip noise
        if msg_type in ("system", "rate_limit_event"):
            return

        if msg_type == "assistant":
            # Real format: { type: "assistant", message: { content: [...] } }
            message = data.get("message", {})
            content_blocks = message.get("content", [])
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")

                    if block_type == "text":
                        text = block.get("text", "").strip()
                        if text:
                            for text_line in text.split("\n"):
                                stripped = text_line.strip()
                                if stripped:
                                    await self.on_log("OPS", stripped)
                                    result_collector.append(stripped)

                    elif block_type == "tool_use":
                        await self._handle_tool_use(block)

        elif msg_type == "tool_use":
            await self._handle_tool_use(data)

        elif msg_type == "tool_result":
            pass  # Tool results are noisy, skip

        elif msg_type == "result":
            # Final result summary
            result_text = data.get("result", "")
            if isinstance(result_text, str) and result_text.strip():
                # Only log if we haven't already from assistant messages
                duration = data.get("duration_ms", 0)
                cost = data.get("total_cost_usd", 0)
                await self.on_log("SYS",
                    f"완료 — {duration / 1000:.1f}초, ${cost:.4f}")
                result_collector.append(result_text.strip())

        elif msg_type == "error":
            error_data = data.get("error", data.get("message", ""))
            if isinstance(error_data, dict):
                await self.on_log("ERR", error_data.get("message", str(error_data)))
            else:
                await self.on_log("ERR", str(error_data))

    async def _handle_tool_use(self, block: dict):
        """Parse a tool_use block and emit a friendly log line."""
        tool_name = block.get("name", "unknown")
        tool_input = block.get("input", {})

        if tool_name == "Write":
            fp = tool_input.get("file_path", "unknown")
            await self.on_log("SYS", f"생성: {_short_path(fp)}")
        elif tool_name == "Edit":
            fp = tool_input.get("file_path", "unknown")
            await self.on_log("SYS", f"수정: {_short_path(fp)}")
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                await self.on_log("SYS", f"$ {cmd[:100]}")
        elif tool_name == "Read":
            fp = tool_input.get("file_path", "unknown")
            await self.on_log("SYS", f"읽기: {_short_path(fp)}")
        elif tool_name == "Glob":
            pattern = tool_input.get("pattern", "")
            await self.on_log("SYS", f"검색: {pattern}")
        elif tool_name == "Grep":
            pattern = tool_input.get("pattern", "")
            await self.on_log("SYS", f"검색: grep {pattern}")
        else:
            await self.on_log("SYS", f"도구: {tool_name}")

    async def run_implementation(self, claude_md_path: str) -> str:
        """Stage 3: Run implementation based on CLAUDE.md."""
        prompt = (
            "프로젝트 루트의 CLAUDE.md를 읽어줘. "
            "이 문서에 정의된 아키텍처와 Task Checklist에 따라 작업을 시작해. "
            "순서: 1) CLAUDE.md의 File Tree대로 디렉토리와 빈 파일 생성 "
            "2) 필요한 패키지/라이브러리 설치 "
            "3) Task Checklist의 순서대로 하나씩 구현 "
            "4) 각 Task 완료 후 간단한 동작 테스트. "
            "작업 중 CLAUDE.md와 충돌하는 부분이 있으면 바로 알려줘. 임의로 설계를 변경하지 마."
        )
        return await self.run_prompt(prompt)

    async def run_implementation_parallel(self, claude_md_path: str, on_log_tagged: Callable,
                                          active_agents: Optional[list[str]] = None,
                                          gating_context: Optional[dict] = None) -> str:
        """Stage 3: Parallel implementation with isolated workspaces.

        Each agent works in its own temp directory, then results are merged
        into the final project directory to prevent race conditions.

        Args:
            active_agents: List of agents to activate (from MoE Gating).
            gating_context: Complexity scores and analysis from Gating Phase.
        """
        import shutil
        import tempfile

        if active_agents is None:
            active_agents = ["fe", "be", "ui"]

        # Read CLAUDE.md
        claude_md = ""
        try:
            with open(claude_md_path, "r", encoding="utf-8") as f:
                claude_md = f.read()
        except Exception:
            await self.on_log("SYS", "CLAUDE.md 읽기 실패, 단일 모드로 실행합니다")
            return await self.run_implementation(claude_md_path)

        agent_labels = {"fe": "Frontend Dev", "be": "Backend Dev", "ui": "UI Designer"}
        active_labels = [agent_labels[a] for a in active_agents]
        await self.on_log("SYS", f"병렬 구현 시작 — 독립 워크스페이스 모드 ({' + '.join(active_labels)})")

        # Create isolated workspace for each agent
        workspaces: dict[str, str] = {}
        for agent_key in active_agents:
            ws = tempfile.mkdtemp(prefix=f"moe-{agent_key}-")
            workspaces[agent_key] = ws
            # Copy CLAUDE.md to each workspace
            shutil.copy2(claude_md_path, os.path.join(ws, "CLAUDE.md"))
            await self.on_log("SYS", f"[{agent_key.upper()}] 워크스페이스 생성: {os.path.basename(ws)}")

        # Agent definitions — each agent has strict boundaries
        agent_prompts = {
            "fe": (
                "프로젝트 루트의 CLAUDE.md를 읽어줘.\n\n"
                "## 너의 역할: Frontend Developer\n"
                "CLAUDE.md의 Task Checklist 중 프론트엔드 관련 Task를 구현해.\n\n"
                "## 담당 범위 (이 파일들만 생성/수정):\n"
                "- src/app/ 또는 src/pages/ — 페이지 컴포넌트 (page.tsx, layout.tsx 포함)\n"
                "- src/components/ — UI 컴포넌트\n"
                "- src/hooks/ — 커스텀 훅\n"
                "- src/store/ 또는 src/context/ — 상태관리\n"
                "- src/lib/ 또는 src/utils/ — 유틸리티\n"
                "- src/types/ — 타입 정의\n"
                "- package.json — 프론트엔드 의존성\n"
                "- tailwind.config, next.config, tsconfig 등 프론트엔드 설정\n\n"
                "## 작업 순서:\n"
                "1) 프로젝트 초기화 (create-next-app 등) — 반드시 현재 디렉토리(.)에서 실행\n"
                "2) 패키지 설치\n"
                "3) 페이지 및 라우팅 구현\n"
                "4) 컴포넌트 구현 (실제 기능 동작하도록)\n"
                "5) 상태관리 연결\n"
                "6) mock 데이터로 동작 확인\n\n"
                "## 절대 금지:\n"
                "- 서브폴더에 새 프로젝트 생성 금지! 반드시 현재 디렉토리(프로젝트 루트)에서 직접 작업할 것\n"
                "- mkdir로 새 프로젝트 폴더를 만들고 그 안에서 create-next-app 하지 마\n"
                "- backend/, api/, server/, db/ 폴더의 파일 생성/수정\n"
                "- globals.css에 디자인 토큰 정의 (UI Designer가 담당)\n"
                "- 임의로 CLAUDE.md 아키텍처 변경"
            ),
            "be": (
                "프로젝트 루트의 CLAUDE.md를 읽어줘.\n\n"
                "## 너의 역할: Backend & Data Engineer\n"
                "CLAUDE.md의 Task Checklist 중 백엔드/DB/API/데이터 관련 Task를 구현해.\n\n"
                "## 담당 범위:\n"
                "### Next.js 프로젝트인 경우:\n"
                "- src/app/api/ — Next.js API Routes\n"
                "- src/services/ — 비즈니스 로직 서비스 레이어\n"
                "- src/lib/supabase.ts, src/lib/db.ts — DB 클라이언트 설정\n"
                "- src/types/database.ts, src/types/api.ts — API/DB 타입 정의\n"
                "- supabase/ — Supabase 마이그레이션, 스키마, 시드 데이터\n"
                "- .env.example — 환경변수 예시 (SUPABASE_URL, API키 등)\n"
                "- src/lib/mockData.ts — 프로토타이핑용 mock 데이터\n\n"
                "### 별도 백엔드인 경우 (FastAPI/Express 등):\n"
                "- backend/ 또는 server/ — 백엔드 코드 전체\n"
                "- models/, schemas/ — 데이터 모델\n"
                "- requirements.txt — 의존성\n\n"
                "## 작업 순서:\n"
                "1) DB 스키마 정의 (Supabase SQL 또는 모델 파일)\n"
                "2) API 엔드포인트 구현 (CRUD)\n"
                "3) 서비스 레이어 구현 (비즈니스 로직)\n"
                "4) mock 데이터 / 시드 데이터 생성\n"
                "5) .env.example 작성\n"
                "6) 타입 정의 (database.ts, api.ts)\n\n"
                "## 중요:\n"
                "- 반드시 현재 디렉토리(프로젝트 루트)에서 작업할 것\n"
                "- FE Agent가 만든 프로젝트 구조(package.json 등)가 이미 있으면 그대로 활용\n"
                "- 프론트엔드가 바로 연결할 수 있도록 API 응답 형식을 타입으로 명시\n\n"
                "## 절대 금지:\n"
                "- src/app/page.tsx, layout.tsx 등 페이지 파일 수정\n"
                "- src/components/ 폴더의 UI 컴포넌트 수정\n"
                "- CSS/스타일 파일 수정\n"
                "- 임의로 CLAUDE.md 아키텍처 변경"
            ),
            "ui": (
                "프로젝트 루트의 CLAUDE.md를 읽어줘.\n\n"
                "## 너의 역할: UI Designer & Stylist\n"
                "CLAUDE.md의 디자인 가이드라인 섹션을 구현해.\n\n"
                "## 담당 범위 (이 파일들만 생성/수정):\n"
                "- src/app/globals.css 또는 src/styles/ — 전역 스타일, 디자인 토큰\n"
                "- src/components/ui/ — 재사용 가능한 기본 UI 컴포넌트 (Button, Card, Input, Modal 등)\n"
                "- tailwind.config — 테마 확장 (색상, 폰트, 스페이싱)\n"
                "- public/ — 파비콘, 로고 등 정적 에셋\n\n"
                "## 작업 순서:\n"
                "1) globals.css에 CSS 변수로 디자인 토큰 정의 (색상, 타이포, 스페이싱, 그림자)\n"
                "2) tailwind.config에 테마 확장\n"
                "3) src/components/ui/ 폴더에 기본 컴포넌트 생성\n"
                "4) 애니메이션/트랜지션 키프레임 정의\n"
                "5) 반응형 브레이크포인트 유틸리티\n\n"
                "## 절대 금지:\n"
                "- page.tsx, layout.tsx 등 페이지 파일 생성/수정 (Frontend Developer가 담당)\n"
                "- 비즈니스 로직, API 호출, 상태관리 코드 작성\n"
                "- backend/ 폴더 접근\n"
                "- 디자인 시스템 '쇼케이스' 페이지 생성 금지 — 컴포넌트만 만들 것"
            ),
        }

        async def run_agent(agent_key: str) -> tuple[str, str]:
            """Run a single agent in its isolated workspace. Returns (agent_key, result)."""
            ws = workspaces[agent_key]
            tag = agent_key.upper()
            agent = ClaudeAgent(ws, lambda p, c: on_log_tagged(tag, p, c))

            # Inject gating context into prompt
            prompt = agent_prompts[agent_key]
            if gating_context and agent_key in gating_context:
                gate = gating_context[agent_key]
                score = gate.get("score", 5)
                mode = gate.get("mode", "normal")
                reasons = gate.get("reasons", [])
                key_tasks = gate.get("key_tasks", [])

                gating_info = f"\n\n## MoE Gating 분석 결과\n"
                gating_info += f"- 복잡도 점수: {score}/10 ({mode} 모드)\n"
                if reasons:
                    gating_info += f"- 주요 이유: {', '.join(reasons[:3])}\n"
                if key_tasks:
                    gating_info += f"- 핵심 작업: {', '.join(key_tasks[:3])}\n"
                if score >= 7:
                    gating_info += "- ⚠️ 복잡도 높음: 특히 위 핵심 작업에 세심한 구현 필요\n"

                prompt += gating_info

            result = await agent.run_prompt(prompt)
            return agent_key, result

        self.is_running = True
        try:
            # Run all active agents in parallel — each in its own workspace
            tasks = [run_agent(k) for k in active_agents]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            results: dict[str, str] = {}
            for r in raw_results:
                if isinstance(r, Exception):
                    await self.on_log("ERR", f"Agent 오류: {r}")
                else:
                    agent_key, output = r
                    results[agent_key] = output

            # ── Merge Phase ──
            # Priority: FE first (owns pages), then BE, then UI (only styles)
            # Files created by higher-priority agent won't be overwritten
            merge_order = [k for k in ["fe", "be", "ui"] if k in active_agents]
            await self.on_log("SYS", f"워크스페이스 병합 시작 (우선순위: {[k.upper() for k in merge_order]})")

            merge_conflicts = 0
            merged_files: set[str] = set()  # Track files already merged (by rel path)

            for agent_key in merge_order:
                ws = workspaces[agent_key]
                tag = agent_key.upper()

                for root, dirs, files in os.walk(ws):
                    dirs[:] = [d for d in dirs if d not in {"node_modules", ".next", "__pycache__", ".git", "venv"}]
                    for fname in files:
                        if fname == "CLAUDE.md":
                            continue
                        src = os.path.join(root, fname)
                        rel = os.path.relpath(src, ws)
                        dst = os.path.join(self.work_dir, rel)

                        # If a higher-priority agent already created this file, skip
                        if rel in merged_files:
                            merge_conflicts += 1
                            await self.on_log("SYS", f"[{tag}] 충돌 스킵: {rel} (이미 상위 Agent가 생성)")
                            continue

                        # Check for conflict with existing project files
                        if os.path.exists(dst):
                            # If file exists, check if it's from another agent
                            try:
                                existing = open(dst, "r", encoding="utf-8", errors="replace").read()
                                new_content = open(src, "r", encoding="utf-8", errors="replace").read()
                                if existing != new_content:
                                    merge_conflicts += 1
                                    await self.on_log("SYS", f"[{tag}] 병합 충돌 감지: {rel} — 덮어씀")
                            except Exception:
                                pass

                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                        merged_files.add(rel)

                await self.on_log("SYS", f"[{tag}] 워크스페이스 병합 완료")

            # Cleanup temp workspaces
            for ws in workspaces.values():
                shutil.rmtree(ws, ignore_errors=True)

            conflict_msg = f" (충돌 {merge_conflicts}건 — 상위 Agent 우선)" if merge_conflicts else ""
            await self.on_log("SYS", f"병렬 구현 완료{conflict_msg}")

            return "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in results.items())

        except Exception as e:
            await self.on_log("ERR", f"병렬 구현 오류: {str(e)}")
            # Cleanup on error
            for ws in workspaces.values():
                shutil.rmtree(ws, ignore_errors=True)
            return ""
        finally:
            self.is_running = False

    async def run_feedback(self, review_content: str) -> str:
        """Stage 5: Apply review feedback based on MoE unified report."""
        prompt = (
            f"MoE 전문가 팀의 코드 리뷰 결과를 반영해서 코드를 수정해줘.\n\n"
            f"## 리뷰 결과:\n{review_content[:8000]}\n\n"
            f"## 수정 우선순위:\n"
            f"1. CRITICAL 항목: 반드시 모두 수정 (보안 취약점, 버그 등)\n"
            f"2. WARNING 항목: 가능한 수정 (설계 개선, 성능 등)\n"
            f"3. SUGGESTION: 무시\n\n"
            f"## 수정 방법:\n"
            f"- Issues 테이블의 Location을 참고해서 해당 파일을 찾아 수정\n"
            f"- Consensus 표시가 있는 이슈는 여러 전문가가 동의한 것이므로 반드시 수정\n"
            f"- 수정 후 빌드/타입 에러가 없는지 확인\n"
            f"- 수정한 파일 목록과 변경 내용을 요약해줘"
        )
        return await self.run_prompt(prompt)

    async def run_custom(self, message: str) -> str:
        """Run a custom prompt."""
        return await self.run_prompt(message)

    async def stop(self):
        """Stop the running process."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.on_log("SYS", "Claude 프로세스 종료됨")
        self.is_running = False


def _short_path(full_path: str) -> str:
    """Shorten file path for display."""
    parts = full_path.replace("\\", "/").split("/")
    if len(parts) <= 3:
        return full_path
    return "/".join(parts[-3:])


_STDERR_IGNORE = [
    "NotOpenSSLWarning",
    "urllib3",
    "warnings.warn",
    "no stdin data received",
    "proceeding without",
    "piping from a slow command",
]


def _is_ignorable_stderr(line: str) -> bool:
    return any(pat in line for pat in _STDERR_IGNORE)

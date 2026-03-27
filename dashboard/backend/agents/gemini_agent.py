"""Gemini CLI agent - manages subprocess interaction with gemini CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime
from typing import Callable, Optional

GEMINI_BIN = shutil.which("gemini") or "gemini"

# Stderr patterns that are warnings, not errors
_STDERR_IGNORE_PATTERNS = [
    # Keytar/keychain warnings
    "keytar", "keychain", "Keychain", "Cannot find module",
    "Require stack", "FileKeychain fallback", "cached credentials",
    "Using File", "node_modules/@google/gemini-cli", "build/Release/",
    # SSL warnings
    "NotOpenSSLWarning", "urllib3",
    # Gemini internal agent/tool messages
    "LocalAgentExecutor", "Skipping subagent", "Blocked call",
    "Unauthorized tool call", "not available to this agent",
    "Tool \"run_shell_command\" not found", "Did you mean one of",
    "Error executing tool", "to prevent recursion",
]


def _is_ignorable_stderr(line: str) -> bool:
    return any(pat in line for pat in _STDERR_IGNORE_PATTERNS)


class GeminiAgent:
    """Controls the Gemini CLI as a subprocess for planning and review tasks."""

    def __init__(self, work_dir: str, on_log: Callable, on_question: Optional[Callable] = None,
                 on_question_structured: Optional[Callable] = None):
        self.work_dir = work_dir
        self.on_log = on_log  # async callback(prefix, content)
        self.on_question = on_question  # async callback() — notify UI that a question is waiting
        self.on_question_structured = on_question_structured  # async callback(question_data) — send structured Q&A
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_running = False
        # For interactive Q&A
        self._user_response: Optional[asyncio.Future] = None
        self._conversation_log: list[str] = []

    async def send_user_response(self, message: str):
        """User sends a response to Gemini's question via the dashboard."""
        if self._user_response and not self._user_response.done():
            self._user_response.set_result(message)

    async def run_prompt(self, prompt: str, save_to: Optional[str] = None, _internal: bool = False) -> str:
        """Run gemini in non-interactive mode with -p flag and stream output."""
        if not _internal:
            self.is_running = True
        await self.on_log("SYS", "Gemini CLI 세션 시작됨")

        cmd = [GEMINI_BIN, "-p", prompt, "-m", "gemini-2.5-flash"]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
                env={**os.environ, "NO_COLOR": "1"},
            )

            output_lines: list[str] = []

            async def read_stream(stream, is_stderr=False):
                buf = ""  # Each stream gets its own buffer
                while True:
                    chunk = await stream.read(256)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    buf += text

                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if is_stderr and _is_ignorable_stderr(line):
                            continue
                        prefix = "ERR" if is_stderr else "GEM"
                        await self.on_log(prefix, line)
                        if not is_stderr:
                            output_lines.append(line)

                if buf.strip():
                    if is_stderr and _is_ignorable_stderr(buf.strip()):
                        return
                    prefix = "ERR" if is_stderr else "GEM"
                    await self.on_log(prefix, buf.strip())
                    if not is_stderr:
                        output_lines.append(buf.strip())

            await asyncio.gather(
                read_stream(self.process.stdout),
                read_stream(self.process.stderr, is_stderr=True),
            )

            await self.process.wait()

            full_output = "\n".join(output_lines)

            if save_to and full_output:
                # Clean Gemini monologue from output before saving
                cleaned = _clean_gemini_output(full_output, save_to)
                save_path = os.path.join(self.work_dir, save_to)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                size_kb = len(cleaned.encode("utf-8")) / 1024
                await self.on_log("SYS", f"{save_to} 저장됨 ({size_kb:.1f}KB)")

            return full_output

        except FileNotFoundError:
            await self.on_log("ERR", f"gemini CLI를 찾을 수 없습니다. 경로: {GEMINI_BIN}")
            return ""
        except Exception as e:
            await self.on_log("ERR", f"Gemini 실행 오류: {str(e)}")
            return ""
        finally:
            if not _internal:
                self.is_running = False
            self.process = None

    async def run_envisioning_interactive(self, user_idea: str, num_questions: int = 4) -> str:
        """Stage 1: Interactive planning with chain-of-questions.

        Gemini asks questions one at a time, user answers via dashboard,
        then Gemini produces final spec.
        """
        self.is_running = True
        self._conversation_log = []

        await self.on_log("SYS", "Gemini CLI 기획 인터뷰 시작")
        await self.on_log("GEM", f"앱 아이디어를 분석합니다: \"{user_idea}\"")

        self._conversation_log.append(f"앱 아이디어: {user_idea}")

        # Step 1: Ask Gemini to generate app-specific feature list for Q3
        await self.on_log("SYS", "핵심 기능 및 타겟 사용자 분석 중...")
        analysis_prompt = f""""{user_idea}" 앱에 대해 아래 두 가지를 JSON으로만 출력하세요. 다른 텍스트 없이 JSON만:
```json
{{
  "features": ["기능1", "기능2", "기능3", "기능4", "기능5", "기능6"],
  "targets": ["타겟1", "타겟2", "타겟3", "타겟4", "타겟5"]
}}
```"""
        analysis_raw = await self.run_prompt(analysis_prompt, _internal=True)

        # Parse features/targets
        features = ["기본 CRUD", "사용자 인증", "대시보드", "검색/필터", "데이터 시각화", "알림"]
        targets = ["학생", "직장인", "크리에이터", "시니어", "전문가"]
        try:
            import re as _re
            json_match = _re.search(r'\{[\s\S]*\}', analysis_raw)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("features"):
                    features = parsed["features"][:8]
                if parsed.get("targets"):
                    targets = parsed["targets"][:6]
        except Exception:
            pass

        # Fixed Q&A structure — structured data, no text parsing needed
        fixed_questions = [
            {
                "id": "q1",
                "text": "어떤 플랫폼으로 만들까요?",
                "options": ["React/Next.js 웹앱", "Flutter 기반 PWA", "React Native 모바일앱"],
                "multi_select": False,
            },
            {
                "id": "q2",
                "text": "어떤 디자인 스타일을 원하시나요?",
                "options": ["Minimal", "Glassmorphism", "Neumorphism", "Brutalist", "Material Design", "다크 모드 중심"],
                "multi_select": False,
            },
            {
                "id": "q3",
                "text": "앱에 포함할 핵심 기능을 선택해주세요",
                "options": features,
                "multi_select": True,
            },
            {
                "id": "q4",
                "text": "타겟 사용자를 선택해주세요",
                "options": targets,
                "multi_select": True,
            },
        ]

        # Step 2: Ask each fixed question via structured event
        for i, q in enumerate(fixed_questions):
            await self.on_log("GEM", f"Q{i+1}: {q['text']}")

            # Send structured question data to frontend
            if self.on_question_structured:
                await self.on_question_structured(q)
            elif self.on_question:
                await self.on_question()

            loop = asyncio.get_running_loop()
            self._user_response = loop.create_future()
            try:
                answer = await asyncio.wait_for(self._user_response, timeout=300)
            except asyncio.TimeoutError:
                answer = q["options"][0] if q["options"] else "적절히 결정해주세요"

            await self.on_log("USR", answer)
            self._conversation_log.append(f"Q: {q['text']}")
            self._conversation_log.append(f"A: {answer}")

        # Step 2.5: Ask for additional instructions (structured)
        extra_q = {
            "id": "extra",
            "text": "추가로 지시하실 사항이 있으신가요?",
            "options": ["예", "아니오"],
            "multi_select": False,
        }
        await self.on_log("GEM", extra_q["text"])

        if self.on_question_structured:
            await self.on_question_structured(extra_q)
        elif self.on_question:
            await self.on_question()

        loop = asyncio.get_running_loop()
        self._user_response = loop.create_future()
        try:
            extra_answer = await asyncio.wait_for(self._user_response, timeout=300)
        except asyncio.TimeoutError:
            extra_answer = "아니오"

        await self.on_log("USR", extra_answer)

        if "예" in extra_answer or "yes" in extra_answer.lower():
            free_q = {
                "id": "extra_text",
                "text": "추가 지시 사항을 입력해주세요.",
                "options": [],
                "multi_select": False,
            }
            await self.on_log("GEM", free_q["text"])

            if self.on_question_structured:
                await self.on_question_structured(free_q)
            elif self.on_question:
                await self.on_question()

            loop = asyncio.get_running_loop()
            self._user_response = loop.create_future()
            try:
                extra_instructions = await asyncio.wait_for(self._user_response, timeout=300)
            except asyncio.TimeoutError:
                extra_instructions = ""

            if extra_instructions.strip():
                await self.on_log("USR", extra_instructions)
                self._conversation_log.append(f"추가 지시: {extra_instructions}")

        # Step 3: Generate final spec based on all answers
        await self.on_log("GEM", "인터뷰 완료. 기획서를 작성합니다...")

        conversation = "\n".join(self._conversation_log)
        spec_prompt = f"""당신은 시니어 프로덕트 매니저 겸 테크 리드입니다.
아래는 사용자와의 인터뷰 내용입니다:

{conversation}

이 인터뷰 내용을 바탕으로 완전한 기획서를 작성해주세요.

아래 형식으로 출력해주세요:

## 프로젝트명: [이름]
## 핵심 목적: [한 줄 설명]
## 타겟 사용자: [설명]
## MVP 기능:
1. [기능1]: [설명]
2. [기능2]: [설명]
## 기술 스택:
- Frontend: [...]
- Backend: [...]
- Database: [...]
- Deployment: [...]
## 데이터 모델:
[엔티티명] - [필드 목록] - [관계]
## 화면 흐름:
Flow 1: [단계1] → [단계2] → [단계3]
## 비기능 요구사항:
- [항목1]"""

        spec = await self.run_prompt(spec_prompt, save_to="docs/01-planning/spec.md", _internal=True)
        self.is_running = False
        return spec

    async def run_envisioning(self, user_idea: str) -> str:
        """Stage 1: Non-interactive fallback — generates spec in one shot."""
        prompt = f"""당신은 시니어 프로덕트 매니저 겸 테크 리드입니다.
사용자가 아래와 같은 앱을 만들고 싶어합니다:

"{user_idea}"

아래 항목을 체계적으로 분석하여 완전한 기획서를 작성해주세요:
1. 핵심 목적 & 타겟 사용자
2. 핵심 기능 목록 (MVP 범위 — 최대 5개)
3. 기술 스택 제안 (프론트/백/DB/배포)
4. 데이터 모델 초안 (주요 엔티티와 관계)
5. 화면 흐름 (메인 유저 플로우 2~3개)
6. 비기능 요구사항 (인증, 반응형, 오프라인 등)

아래 형식으로 출력해주세요:

## 프로젝트명: [이름]
## 핵심 목적: [한 줄 설명]
## 타겟 사용자: [설명]
## MVP 기능:
1. [기능1]: [설명]
2. [기능2]: [설명]
## 기술 스택:
- Frontend: [...]
- Backend: [...]
- Database: [...]
- Deployment: [...]
## 데이터 모델:
[엔티티명] - [필드 목록] - [관계]
## 화면 흐름:
Flow 1: [단계1] → [단계2] → [단계3]
## 비기능 요구사항:
- [항목1]"""

        return await self.run_prompt(prompt, save_to="docs/01-planning/spec.md")

    async def run_blueprinting(self, spec_content: str) -> str:
        """Stage 2: Generate CLAUDE.md from spec."""
        prompt = f"""아래는 확정된 앱 기획서입니다:

{spec_content}

이 기획서를 바탕으로 'CLAUDE.md' 파일을 작성해 주세요.
Claude Code CLI가 이 파일을 읽고 자율적으로 코딩할 수 있어야 합니다.

CLAUDE.md에 반드시 포함할 내용:
### 1. 프로젝트 개요 — 앱 이름, 목적, 타겟 사용자 (1~2문장)
### 2. 기술 스택 (버전 포함) — 각 기술의 선택 이유 한 줄씩
  - 기획서에 명시된 프레임워크/플랫폼을 반드시 따를 것 (Flutter, React, Next.js 등)
  - Flutter 기반인 경우: Flutter Web + PWA 설정 포함, dart 패키지 명시
### 3. 프로젝트 구조 (File Tree) — 각 디렉토리의 역할 설명
### 4. 데이터 모델 상세 — 각 엔티티의 필드, 타입, 관계를 코드 수준으로
### 5. 구현 우선순위 (Task Checklist)
- [ ] Task 1: [설명] — 예상 파일: [파일명]
- [ ] Task 2: ...
### 6. 코딩 컨벤션 — 네이밍 규칙, 폴더 구조 원칙, 에러 처리 패턴
### 7. 디자인 가이드라인 (매우 중요!)
  - 기획서에 명시된 디자인 스타일을 반드시 구체적으로 반영
  - 포함할 내용: 컬러 팔레트 (hex 코드), 타이포그래피, 컴포넌트 스타일링 규칙
  - UI 라이브러리 지정 시 해당 라이브러리 사용법 명시
  - 다크모드 요구 시 다크/라이트 테마 변수 정의
  - 애니메이션/트랜지션 가이드라인
  - 반응형 브레이크포인트 정의
### 8. AI 연동 가이드라인 (프로토타이핑용)
  - 앱에서 AI 기능이 필요한 경우 (챗봇, 텍스트 분석, 추천 등):
    - .env 파일에 GOOGLE_API_KEY 환경변수를 사용하도록 구현
    - Gemini API (https://generativelanguage.googleapis.com/v1beta/) 활용
    - API 키가 없을 때도 앱이 동작하도록 fallback/mock 데이터 제공
    - API 키는 절대 코드에 하드코딩하지 말 것
  - AI 기능이 불필요한 앱이면 이 섹션은 "해당 없음"으로 표시
### 9. 테스트 요구사항 — 단위 테스트 필수 대상, 테스트 프레임워크
### 10. 금지 사항 — 하지 말아야 할 것들"""

        return await self.run_prompt(prompt, save_to="CLAUDE.md")

    async def run_review(self, code_summary: str) -> str:
        """Stage 4: Single expert code review (fallback)."""
        prompt = f"""아래는 구현된 코드의 요약입니다:

{code_summary}

다음 관점에서 코드 리뷰를 수행해 주세요:
1. 설계 준수 여부
2. 로직 오류
3. 보안 취약점
4. 성능 이슈
5. 코드 품질

리뷰 결과를 아래 형식으로 출력해 주세요:

## 코드 리뷰 리포트
### 🔴 Critical (즉시 수정 필요)
- [파일명:라인] 문제 설명 → 수정 제안
### 🟡 Warning (권장 수정)
- [파일명:라인] 문제 설명 → 수정 제안
### 🟢 Suggestion (개선 아이디어)
- [파일명:라인] 제안 내용
### 📊 종합 점수: [1~10] / 10"""

        return await self.run_prompt(prompt, save_to="docs/04-reviews/review.md")

    async def run_review_moe(self, code_summary: str, experts: Optional[list[dict]] = None) -> str:
        """Stage 4: MoE parallel review — selected experts review simultaneously.

        Args:
            experts: List of expert dicts from MoE Gating. If None, uses all 5.
        """
        if experts is None:
            experts = [
                {"name": "아키텍처 전문가", "focus": "시스템 설계, 컴포넌트 분리, 확장성", "prefix": "ARCH"},
                {"name": "보안 전문가", "focus": "인증/인가, XSS, SQL Injection, OWASP Top 10", "prefix": "SEC"},
                {"name": "성능 전문가", "focus": "리렌더링, N+1 쿼리, 번들 크기, 캐싱", "prefix": "PERF"},
                {"name": "코드 품질 전문가", "focus": "네이밍, 중복, 타입 안정성, 가독성", "prefix": "QUAL"},
                {"name": "UX/디자인 전문가", "focus": "반응형, 접근성, 로딩/에러 상태", "prefix": "UX"},
            ]

        self.is_running = True
        await self.on_log("SYS", f"MoE 병렬 리뷰 시작 ({len(experts)}명의 전문가)")

        async def run_expert(expert: dict) -> str:
            prompt = f"""당신은 {expert['name']}입니다.
전문 분야: {expert['focus']}

아래 코드를 당신의 전문 분야 관점에서만 리뷰해주세요.
다른 분야는 다른 전문가가 담당하니 당신의 전문 분야에만 집중하세요.
이슈가 없으면 빈 배열로 출력하세요. 거짓 이슈를 만들지 마세요.

{code_summary}

결과를 반드시 아래 JSON 형식으로만 출력하세요 (다른 텍스트 없이 JSON만):

```json
{{
  "expert": "{expert['prefix']}",
  "name": "{expert['name']}",
  "score": 8,
  "confidence": 0.8,
  "issues": [
    {{
      "severity": "CRITICAL 또는 WARNING 또는 SUGGESTION",
      "location": "파일명:라인 또는 섹션명",
      "title": "이슈 제목",
      "description": "상세 설명",
      "suggestion": "수정 제안"
    }}
  ],
  "praise": ["잘한 점 1", "잘한 점 2"]
}}
```"""

            agent = GeminiAgent(self.work_dir, self.on_log)
            return await agent.run_prompt(prompt, _internal=True)

        # Run experts sequentially with delay (avoid 429 rate limit)
        results = []
        for expert in experts:
            await self.on_log("GEM", f"[{expert['prefix']}] {expert['name']} 리뷰 시작")
            try:
                result = await run_expert(expert)
                results.append(result)
            except Exception as e:
                results.append(e)
            await asyncio.sleep(3)  # Rate limit delay

        # ── Parse & Integrate Results ──
        parsed_reviews: list[dict] = []
        raw_texts: list[str] = []

        for i, result in enumerate(results):
            prefix = experts[i]["prefix"]
            name = experts[i]["name"]

            if isinstance(result, Exception):
                await self.on_log("ERR", f"[{prefix}] {name} 오류: {result}")
                continue

            await self.on_log("GEM", f"[{prefix}] {name} 리뷰 완료")
            raw_texts.append(result)

            # Try to parse JSON from the result
            parsed = _parse_expert_json(result)
            if parsed:
                parsed_reviews.append(parsed)
            else:
                # Fallback: treat as markdown
                parsed_reviews.append({
                    "expert": prefix,
                    "name": name,
                    "score": 0,
                    "confidence": 0.5,
                    "issues": [],
                    "praise": [],
                    "raw": result,
                })

        # ── Generate Unified Report ──
        full_review = _generate_unified_report(parsed_reviews, raw_texts)

        # Save
        save_path = os.path.join(self.work_dir, "docs/04-reviews/review.md")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(full_review)
        await self.on_log("SYS", f"docs/04-reviews/review.md 저장됨 ({len(full_review.encode('utf-8')) / 1024:.1f}KB)")

        # Log summary
        total_issues = sum(len(r.get("issues", [])) for r in parsed_reviews)
        critical_count = sum(1 for r in parsed_reviews for iss in r.get("issues", []) if iss.get("severity", "").upper() == "CRITICAL")
        avg_score = 0
        scored = [r for r in parsed_reviews if r.get("score", 0) > 0]
        if scored:
            total_weight = sum(r.get("confidence", 0.5) for r in scored)
            if total_weight > 0:
                avg_score = sum(r["score"] * r.get("confidence", 0.5) for r in scored) / total_weight

        await self.on_log("SYS",
            f"통합 리뷰: {len(parsed_reviews)}명 완료 | "
            f"이슈 {total_issues}건 (Critical {critical_count}) | "
            f"가중 평균 점수: {avg_score:.1f}/10")

        self.is_running = False
        return full_review

    async def stop(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            await self.on_log("SYS", "Gemini 프로세스 종료됨")
        self.is_running = False
        if self._user_response and not self._user_response.done():
            self._user_response.cancel()


# ---------------------------------------------------------------------------
# Review result parsing & integration helpers
# ---------------------------------------------------------------------------


def _clean_gemini_output(text: str, filename: str) -> str:
    """Remove Gemini's internal monologue from output.

    Gemini often prepends "I will read the file...", "I will now create..."
    before the actual content. This extracts only the meaningful content.
    """
    import re

    # Strategy 1: If there's a markdown code block, extract its content
    md_block = re.search(r'```(?:markdown)?\s*\n([\s\S]+?)\n```', text)
    if md_block:
        content = md_block.group(1).strip()
        if len(content) > 100:  # Only use if substantial
            return content

    # Strategy 2: Find the first markdown heading (# ) and take everything from there
    heading_match = re.search(r'^(#{1,3}\s+.+)$', text, re.MULTILINE)
    if heading_match:
        idx = text.index(heading_match.group(0))
        content = text[idx:].strip()
        if len(content) > 100:
            return content

    # Strategy 3: Remove known Gemini monologue patterns from the beginning
    lines = text.split("\n")
    clean_lines = []
    found_content = False
    monologue_patterns = [
        r'^I will',
        r'^I\'ll',
        r'^Let me',
        r'^Now I',
        r'^기획서를 바탕으로',
        r'^현재 환경',
        r'^제공해 드립니다',
        r'^아래에 전체',
        r'^이 내용을',
        r'^저장해 주세요',
        r'^작업을 시작',
        r'^먼저',
    ]

    for line in lines:
        stripped = line.strip()
        if not found_content:
            if any(re.match(p, stripped) for p in monologue_patterns):
                continue
            if not stripped:
                continue
            found_content = True
        clean_lines.append(line)

    return "\n".join(clean_lines).strip() if clean_lines else text


def _parse_expert_json(raw_text: str) -> Optional[dict]:
    """Try to extract JSON from expert review output."""
    import re

    # Try to find JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*\n({[\s\S]*?})\s*\n```', raw_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r'(\{[\s\S]*"expert"[\s\S]*\})', raw_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _generate_unified_report(parsed_reviews: list[dict], raw_texts: list[str]) -> str:
    """Generate a unified review report from parsed expert reviews.

    Includes:
    - Expert verdicts table
    - Issues sorted by severity with deduplication
    - Weighted average score
    - Consensus analysis
    """
    lines = ["# MoE Unified Review Report\n"]

    # ── Expert Verdicts Table ──
    lines.append("## Expert Verdicts\n")
    lines.append("| Expert | Score | Confidence | Issues | Verdict |")
    lines.append("|--------|-------|------------|--------|---------|")

    for r in parsed_reviews:
        prefix = r.get("expert", "?")
        name = r.get("name", "Unknown")
        score = r.get("score", 0)
        confidence = r.get("confidence", 0.5)
        issues = r.get("issues", [])
        critical = sum(1 for i in issues if i.get("severity", "").upper() == "CRITICAL")
        verdict = "APPROVED" if critical == 0 and score >= 7 else "CHANGES_REQUESTED"
        emoji = "✅" if verdict == "APPROVED" else "🔴"

        lines.append(f"| [{prefix}] {name} | {score}/10 | {confidence:.1f} | {len(issues)} | {emoji} {verdict} |")

    # ── Weighted Average ──
    scored = [r for r in parsed_reviews if r.get("score", 0) > 0]
    if scored:
        total_weight = sum(r.get("confidence", 0.5) for r in scored)
        avg = sum(r["score"] * r.get("confidence", 0.5) for r in scored) / total_weight if total_weight else 0
        lines.append(f"\n**Weighted Average Score: {avg:.1f} / 10**\n")

    # ── All Issues (sorted by severity) ──
    all_issues: list[tuple[str, dict]] = []
    for r in parsed_reviews:
        prefix = r.get("expert", "?")
        for issue in r.get("issues", []):
            all_issues.append((prefix, issue))

    # Sort: CRITICAL > WARNING > SUGGESTION
    severity_order = {"CRITICAL": 0, "WARNING": 1, "SUGGESTION": 2}
    all_issues.sort(key=lambda x: severity_order.get(x[1].get("severity", "").upper(), 3))

    # Deduplicate by similar titles
    seen_titles: set[str] = set()
    unique_issues: list[tuple[str, dict, int]] = []  # (prefix, issue, consensus_count)

    for prefix, issue in all_issues:
        title = issue.get("title", "").lower().strip()
        # Simple dedup: check if any similar title exists
        is_dup = False
        for seen in seen_titles:
            if title and (title in seen or seen in title or
                         len(set(title.split()) & set(seen.split())) >= 3):
                # Find the existing issue and bump consensus
                for j, (_, _, count) in enumerate(unique_issues):
                    if unique_issues[j][1].get("title", "").lower().strip() == seen:
                        unique_issues[j] = (unique_issues[j][0], unique_issues[j][1], count + 1)
                        break
                is_dup = True
                break
        if not is_dup:
            seen_titles.add(title)
            unique_issues.append((prefix, issue, 1))

    if unique_issues:
        lines.append("\n## Issues (Priority Sorted)\n")
        lines.append("| # | Severity | Location | Issue | Flagged By | Consensus |")
        lines.append("|---|----------|----------|-------|-----------|-----------|")

        severity_emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "SUGGESTION": "🟢"}
        for idx, (prefix, issue, consensus) in enumerate(unique_issues, 1):
            sev = issue.get("severity", "?").upper()
            emoji = severity_emoji.get(sev, "⚪")
            loc = issue.get("location", "-")
            title = issue.get("title", "-")
            consensus_str = f"x{consensus}" if consensus > 1 else ""
            lines.append(f"| {idx} | {emoji} {sev} | `{loc}` | {title} | [{prefix}] | {consensus_str} |")

        # Suggestions
        lines.append("")
        for _, issue, _ in unique_issues:
            if issue.get("suggestion"):
                lines.append(f"- **{issue.get('title', '')}**: {issue['suggestion']}")

    # ── Praise ──
    all_praise = []
    for r in parsed_reviews:
        for p in r.get("praise", []):
            if p and p not in all_praise:
                all_praise.append(p)

    if all_praise:
        lines.append("\n## What Was Done Well\n")
        for p in all_praise:
            lines.append(f"- {p}")

    # ── Statistics ──
    total = len(all_issues)
    critical = sum(1 for _, i in all_issues if i.get("severity", "").upper() == "CRITICAL")
    warning = sum(1 for _, i in all_issues if i.get("severity", "").upper() == "WARNING")
    suggestion = sum(1 for _, i in all_issues if i.get("severity", "").upper() == "SUGGESTION")

    lines.append("\n## Statistics\n")
    lines.append(f"- Experts: {len(parsed_reviews)}")
    lines.append(f"- Total issues: {total} (CRITICAL {critical}, WARNING {warning}, SUGGESTION {suggestion})")
    lines.append(f"- Unique issues (after dedup): {len(unique_issues)}")

    # ── Raw outputs as appendix ──
    lines.append("\n---\n\n## Appendix: Raw Expert Outputs\n")
    for i, text in enumerate(raw_texts):
        lines.append(f"<details><summary>Expert {i+1} Raw Output</summary>\n")
        lines.append(f"```\n{text[:3000]}\n```\n")
        lines.append("</details>\n")

    return "\n".join(lines)

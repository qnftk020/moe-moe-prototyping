"""Planner Agent — uses Claude Code CLI with Sonnet model for planning and review."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Callable, Optional

CLAUDE_BIN = shutil.which("claude") or "claude"


class PlannerAgent:
    """Claude Sonnet-based agent for planning (Stage 1-2) and review (Stage 4).

    Uses --model sonnet for faster, cheaper planning/review tasks.
    Same interface as GeminiAgent for drop-in replacement.
    """

    def __init__(self, work_dir: str, on_log: Callable, on_question: Optional[Callable] = None,
                 on_question_structured: Optional[Callable] = None):
        self.work_dir = work_dir
        self.on_log = on_log
        self.on_question = on_question
        self.on_question_structured = on_question_structured
        self.process: Optional[asyncio.subprocess.Process] = None
        self.is_running = False
        self._user_response: Optional[asyncio.Future] = None
        self._conversation_log: list[str] = []

    async def send_user_response(self, message: str):
        if self._user_response and not self._user_response.done():
            self._user_response.set_result(message)

    async def run_prompt(self, prompt: str, save_to: Optional[str] = None, _internal: bool = False) -> str:
        """Run Claude Code with Sonnet model in non-interactive mode."""
        if not _internal:
            self.is_running = True
        await self.on_log("SYS", "Claude Sonnet 세션 시작됨")

        cmd = [
            CLAUDE_BIN,
            "-p", prompt,
            "--model", "sonnet",
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

            output_lines: list[str] = []
            buf = ""

            async def read_stdout():
                nonlocal buf
                while True:
                    chunk = await self.process.stdout.read(4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    buf += text
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        await self._process_line(line, output_lines)
                if buf.strip():
                    await self._process_line(buf.strip(), output_lines)

            async def read_stderr():
                while True:
                    chunk = await self.process.stderr.read(256)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace").strip()
                    if text and not _is_ignorable(text):
                        await self.on_log("ERR", text)

            await asyncio.gather(read_stdout(), read_stderr())
            await self.process.wait()

            full_output = "\n".join(output_lines)

            if save_to and full_output:
                cleaned = _clean_output(full_output)
                save_path = os.path.join(self.work_dir, save_to)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                size_kb = len(cleaned.encode("utf-8")) / 1024
                await self.on_log("SYS", f"{save_to} 저장됨 ({size_kb:.1f}KB)")

            return full_output

        except FileNotFoundError:
            await self.on_log("ERR", f"claude CLI를 찾을 수 없습니다")
            return ""
        except Exception as e:
            await self.on_log("ERR", f"Claude Sonnet 오류: {str(e)}")
            return ""
        finally:
            if not _internal:
                self.is_running = False
            self.process = None

    async def _process_line(self, line: str, collector: list[str]):
        """Process a stream-json line from Claude."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            if line.strip():
                await self.on_log("SNT", line)
                collector.append(line)
            return

        msg_type = data.get("type", "")

        if msg_type in ("system", "rate_limit_event"):
            return

        if msg_type == "assistant":
            message = data.get("message", {})
            content_blocks = message.get("content", [])
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            for text_line in text.split("\n"):
                                stripped = text_line.strip()
                                if stripped:
                                    await self.on_log("SNT", stripped)
                                    collector.append(stripped)

        elif msg_type == "result":
            duration = data.get("duration_ms", 0)
            cost = data.get("total_cost_usd", 0)
            await self.on_log("SYS", f"완료 — {duration / 1000:.1f}초")
            result_text = data.get("result", "")
            if isinstance(result_text, str) and result_text.strip():
                collector.append(result_text.strip())

        elif msg_type == "error":
            error_data = data.get("error", data.get("message", ""))
            if isinstance(error_data, dict):
                await self.on_log("ERR", error_data.get("message", str(error_data)))
            else:
                await self.on_log("ERR", str(error_data))

    # ── Stage 1: Interactive Q&A ──

    async def run_envisioning_interactive(self, user_idea: str, num_questions: int = 4) -> str:
        self.is_running = True
        self._conversation_log = []

        await self.on_log("SYS", "Claude Sonnet 기획 인터뷰 시작")
        await self.on_log("SNT", f"앱 아이디어를 분석합니다: \"{user_idea}\"")

        self._conversation_log.append(f"앱 아이디어: {user_idea}")

        # Generate app-specific features/targets
        await self.on_log("SYS", "핵심 기능 및 타겟 사용자 분석 중...")
        analysis_prompt = f""""{user_idea}" 앱에 대해 아래 두 가지를 JSON으로만 출력하세요. 다른 텍스트 없이 JSON만:
```json
{{
  "features": ["기능1", "기능2", "기능3", "기능4", "기능5", "기능6"],
  "targets": ["타겟1", "타겟2", "타겟3", "타겟4", "타겟5"]
}}
```"""
        analysis_raw = await self.run_prompt(analysis_prompt, _internal=True)

        features = ["기본 CRUD", "사용자 인증", "대시보드", "검색/필터", "데이터 시각화", "알림"]
        targets = ["학생", "직장인", "크리에이터", "시니어", "전문가"]
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis_raw)
            if json_match:
                parsed = json.loads(json_match.group())
                if parsed.get("features"):
                    features = parsed["features"][:8]
                if parsed.get("targets"):
                    targets = parsed["targets"][:6]
        except Exception:
            pass

        # Fixed Q&A structure
        fixed_questions = [
            {"id": "q1", "text": "어떤 플랫폼으로 만들까요?",
             "options": ["React/Next.js 웹앱", "Flutter 기반 PWA", "React Native 모바일앱"], "multi_select": False},
            {"id": "q2", "text": "어떤 디자인 스타일을 원하시나요?",
             "options": ["Minimal", "Glassmorphism", "Neumorphism", "Brutalist", "Material Design", "다크 모드 중심"], "multi_select": False},
            {"id": "q3", "text": "앱에 포함할 핵심 기능을 선택해주세요",
             "options": features, "multi_select": True},
            {"id": "q4", "text": "타겟 사용자를 선택해주세요",
             "options": targets, "multi_select": True},
        ]

        for i, q in enumerate(fixed_questions):
            await self.on_log("SNT", f"Q{i+1}: {q['text']}")
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

        # Additional instructions
        extra_q = {"id": "extra", "text": "추가로 지시하실 사항이 있으신가요?",
                   "options": ["예", "아니오"], "multi_select": False}
        await self.on_log("SNT", extra_q["text"])
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
            free_q = {"id": "extra_text", "text": "추가 지시 사항을 입력해주세요.", "options": [], "multi_select": False}
            await self.on_log("SNT", free_q["text"])
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

        # Generate spec
        await self.on_log("SNT", "인터뷰 완료. 기획서를 작성합니다...")
        conversation = "\n".join(self._conversation_log)
        spec_prompt = f"""당신은 10년 경력의 시니어 프로덕트 매니저입니다. 아래 사용자 인터뷰를 바탕으로 **매우 상세한** 앱 기획서를 작성하세요.

모든 섹션을 구체적으로, 실제 개발자가 바로 구현할 수 있을 수준으로 작성하세요. 추상적이거나 "등"으로 끝나는 설명은 금지합니다.

{conversation}

아래 형식을 정확히 따르되, 각 항목을 최대한 상세하게 작성하세요:

## 프로젝트명: [창의적이고 기억하기 쉬운 이름]

## 핵심 목적
[2~3문장으로 앱이 해결하는 문제와 제공하는 가치를 명확히 서술]

## 타겟 사용자
- 주요 타겟: [구체적 페르소나 — 나이, 직업, 상황]
- 보조 타겟: [두 번째 사용자군]
- 사용 시나리오: [언제, 어디서, 왜 이 앱을 쓰는지]

## MVP 기능 (각 기능을 3~4줄로 상세 설명)
1. **[기능명]**: [상세 설명 — 사용자가 무엇을 할 수 있는지, 어떤 UI가 필요한지, 데이터는 어떻게 처리되는지]
2. **[기능명]**: [상세 설명]
3. **[기능명]**: [상세 설명]
4. **[기능명]**: [상세 설명]
5. **[기능명]**: [상세 설명]

## 기술 스택 (각 기술의 선택 이유 포함)
- **Frontend**: [프레임워크 + 버전] — [선택 이유]
- **Backend**: [프레임워크 + 버전] — [선택 이유]
- **Database**: [DB + 호스팅] — [선택 이유]
- **상태관리**: [라이브러리] — [선택 이유]
- **스타일링**: [CSS 프레임워크 + UI 라이브러리] — [선택 이유]
- **인증**: [방식] — [선택 이유]
- **배포**: [플랫폼] — [선택 이유]

## 데이터 모델 (각 엔티티의 필드를 타입까지 명시)
### User
- id: UUID (PK)
- email: string (unique)
- [나머지 필드: 타입]
- 관계: [다른 엔티티와의 관계]

### [엔티티2]
- [필드: 타입]
- 관계: [관계]

(모든 엔티티를 나열)

## 화면 흐름 (각 화면의 UI 요소 명시)
### Flow 1: [플로우명]
1. **[화면명]** — [화면 설명, 포함 UI 요소: 버튼/입력/리스트 등]
2. **[화면명]** — [화면 설명]
3. **[화면명]** — [화면 설명]

### Flow 2: [플로우명]
1. **[화면명]** — [화면 설명]
2. **[화면명]** — [화면 설명]

### Flow 3: [플로우명]
1. **[화면명]** — [화면 설명]

## 디자인 요구사항
- **스타일**: [선택된 디자인 스타일의 구체적 특징]
- **컬러 팔레트**: Primary [hex], Secondary [hex], Background [hex], Text [hex], Accent [hex]
- **타이포그래피**: 제목 [폰트/크기], 본문 [폰트/크기]
- **다크/라이트 모드**: [지원 여부 및 방식]
- **애니메이션**: [어떤 인터랙션에 어떤 애니메이션]
- **반응형**: [모바일/태블릿/데스크탑 각각의 레이아웃 차이]

## 비기능 요구사항
- **인증**: [방식 상세]
- **반응형**: [브레이크포인트]
- **오프라인**: [지원 여부 및 방식]
- **성능**: [목표 지표]
- **접근성**: [a11y 요구사항]
- **보안**: [보안 요구사항]"""

        spec = await self.run_prompt(spec_prompt, save_to="docs/01-planning/spec.md", _internal=True)
        self.is_running = False
        return spec

    # ── Stage 2: Blueprinting ──

    async def run_blueprinting(self, spec_content: str) -> str:
        prompt = f"""당신은 시니어 테크 리드입니다. 아래 기획서를 바탕으로 Claude Code CLI가 자율적으로 코딩할 수 있는 **매우 상세한** CLAUDE.md를 작성하세요.

이 문서를 읽는 AI 코딩 에이전트가 추가 질문 없이 바로 구현을 시작할 수 있어야 합니다. 모호한 설명은 금지합니다.

{spec_content}

아래 모든 섹션을 빠짐없이, 구체적으로 작성하세요:

### 1. 프로젝트 개요
- 앱 이름, 목적, 타겟 사용자 (2~3문장)

### 2. 기술 스택 (버전 반드시 포함)
각 기술의 선택 이유를 한 줄씩 작성:
- Frontend: [이름 버전] — [이유]
- Backend: [이름 버전] — [이유]
- Database: [이름] — [이유]
- Styling: [프레임워크] — [이유]
- State Management: [라이브러리] — [이유]
- Auth: [방식] — [이유]

### 3. 프로젝트 구조 (File Tree)
모든 파일과 폴더를 나열하고, 각 디렉토리의 역할을 한 줄로 설명:
```
project-name/
├── src/
│   ├── app/
│   │   ├── page.tsx          # 메인 홈 페이지 — [구체적 설명]
│   │   ├── layout.tsx        # 루트 레이아웃 — [설명]
│   │   ├── globals.css       # 글로벌 스타일, CSS 변수
│   │   └── [페이지명]/
│   │       └── page.tsx      # [페이지 설명]
│   ├── components/
│   │   ├── [컴포넌트명].tsx  # [역할 설명]
│   │   └── ui/              # 재사용 UI 컴포넌트
│   ├── hooks/               # 커스텀 훅
│   ├── lib/                 # 유틸리티, API 클라이언트
│   ├── store/               # 상태관리
│   └── types/               # 타입 정의
├── public/                  # 정적 에셋
└── package.json
```

### 4. 데이터 모델 상세
각 엔티티를 TypeScript 인터페이스 수준으로 정의:
```typescript
interface User {{
  id: string;          // UUID
  email: string;       // 로그인용 이메일
  name: string;        // 표시 이름
  // ... 모든 필드
}}
```
엔티티 간 관계도 명시 (1:N, N:M 등)

### 5. 구현 우선순위 (Task Checklist)
순서대로, 각 Task에 대상 파일명과 구체적 구현 내용 명시:
- [ ] Task 1: [구체적 설명] — 파일: [파일명1, 파일명2]
  - 세부: [무엇을 구현하는지 2~3줄로]
- [ ] Task 2: [설명] — 파일: [파일명]
  - 세부: [구현 내용]
(최소 7개 이상의 Task)

### 6. 코딩 컨벤션
- 네이밍: [변수, 함수, 컴포넌트, 파일 네이밍 규칙]
- 폴더 구조: [원칙]
- 에러 처리: [패턴]
- 컴포넌트 작성: [규칙 — props 타입, 기본값 등]

### 7. 디자인 가이드라인 (매우 상세하게!)
#### 컬러 팔레트 (CSS 변수로 정의)
```css
:root {{
  --color-primary: #[hex];
  --color-secondary: #[hex];
  --color-background: #[hex];
  --color-surface: #[hex];
  --color-text-primary: #[hex];
  --color-text-secondary: #[hex];
  --color-accent: #[hex];
  --color-danger: #[hex];
  --color-success: #[hex];
  --color-border: #[hex];
}}
```

#### 타이포그래피
- 제목: [폰트패밀리, 크기, 굵기]
- 본문: [폰트패밀리, 크기]
- 캡션: [크기]

#### 간격 (Spacing)
- xs: 4px, sm: 8px, md: 16px, lg: 24px, xl: 32px

#### 라운딩 (Border Radius)
- sm: 4px, md: 8px, lg: 12px, full: 9999px

#### 그림자 (Box Shadow)
- sm: [값], md: [값], lg: [값]

#### 애니메이션
- hover: [transition 값]
- 페이지 전환: [방식]
- 로딩: [방식]

#### 반응형 브레이크포인트
- mobile: < 640px
- tablet: 640px ~ 1024px
- desktop: > 1024px

### 8. 각 페이지 상세 (화면별 구현 가이드)
#### 메인 페이지 (/)
- 레이아웃: [구체적 배치 설명]
- 포함 컴포넌트: [컴포넌트 목록]
- 기능: [사용자가 할 수 있는 것]
- 데이터: [어떤 데이터를 보여주는지]

#### [두 번째 페이지]
- [같은 형식으로]

(모든 페이지에 대해)

### 9. Mock 데이터
프로토타이핑을 위한 mock 데이터를 구체적으로 정의:
```typescript
export const mockUsers = [
  {{ id: "1", name: "홍길동", ... }},
];
```

### 10. 금지 사항
- 외부 API 키 하드코딩 금지
- any 타입 사용 금지
- console.log 남기기 금지
- 인라인 스타일 금지 (Tailwind 사용)
- 미사용 import 남기기 금지"""

        return await self.run_prompt(prompt, save_to="CLAUDE.md")

    # ── Stage 4: MoE Review ──

    async def run_review_moe(self, code_summary: str, experts: Optional[list[dict]] = None) -> str:
        if experts is None:
            experts = [
                {"name": "아키텍처 전문가", "focus": "시스템 설계, 확장성", "prefix": "ARCH"},
                {"name": "보안 전문가", "focus": "인증, XSS, OWASP Top 10", "prefix": "SEC"},
                {"name": "성능 전문가", "focus": "리렌더링, 캐싱, 번들", "prefix": "PERF"},
                {"name": "코드 품질 전문가", "focus": "네이밍, 중복, 타입", "prefix": "QUAL"},
                {"name": "UX 전문가", "focus": "반응형, 접근성, 상태 처리", "prefix": "UX"},
            ]

        self.is_running = True
        await self.on_log("SYS", f"MoE 리뷰 시작 ({len(experts)}명 — Claude Sonnet 순차 실행)")

        results = []
        for expert in experts:
            await self.on_log("SNT", f"[{expert['prefix']}] {expert['name']} 리뷰 시작")
            prompt = f"""당신은 {expert['name']}입니다. 전문 분야: {expert['focus']}

아래 코드를 당신의 전문 분야에서만 리뷰하세요. JSON으로만 출력:

```json
{{
  "expert": "{expert['prefix']}",
  "name": "{expert['name']}",
  "score": 8,
  "confidence": 0.8,
  "issues": [
    {{"severity": "CRITICAL 또는 WARNING 또는 SUGGESTION", "location": "파일:라인", "title": "제목", "description": "설명", "suggestion": "수정 제안"}}
  ],
  "praise": ["잘한 점"]
}}
```

{code_summary[:6000]}"""

            try:
                result = await self.run_prompt(prompt, _internal=True)
                results.append((expert, result))
            except Exception as e:
                results.append((expert, str(e)))
            await asyncio.sleep(2)  # Rate limit delay

        # Build unified report
        from agents.gemini_agent import _parse_expert_json, _generate_unified_report
        parsed_reviews = []
        raw_texts = []
        for expert, result in results:
            raw_texts.append(result if isinstance(result, str) else "")
            parsed = _parse_expert_json(result) if isinstance(result, str) else None
            if parsed:
                parsed_reviews.append(parsed)
            else:
                parsed_reviews.append({"expert": expert["prefix"], "name": expert["name"],
                                       "score": 0, "confidence": 0.5, "issues": [], "praise": []})
            await self.on_log("SNT", f"[{expert['prefix']}] {expert['name']} 리뷰 완료")

        full_review = _generate_unified_report(parsed_reviews, raw_texts)

        save_path = os.path.join(self.work_dir, "docs/04-reviews/review.md")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(full_review)
        await self.on_log("SYS", f"docs/04-reviews/review.md 저장됨")

        self.is_running = False
        return full_review

    async def stop(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
        self.is_running = False
        if self._user_response and not self._user_response.done():
            self._user_response.cancel()


def _is_ignorable(line: str) -> bool:
    patterns = ["NotOpenSSLWarning", "urllib3", "warnings.warn",
                "no stdin data received", "proceeding without", "piping from"]
    return any(p in line for p in patterns)


def _clean_output(text: str) -> str:
    """Remove internal monologue from output."""
    import re
    md_block = re.search(r'```(?:markdown)?\s*\n([\s\S]+?)\n```', text)
    if md_block and len(md_block.group(1).strip()) > 100:
        return md_block.group(1).strip()
    heading = re.search(r'^(#{1,3}\s+.+)$', text, re.MULTILINE)
    if heading:
        idx = text.index(heading.group(0))
        content = text[idx:].strip()
        if len(content) > 100:
            return content
    return text

"""Stitch 2.0 MCP agent — uses Google Stitch to generate UI designs."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Callable, Optional


class StitchAgent:
    """Controls Google Stitch 2.0 via MCP CLI to generate UI designs."""

    def __init__(self, work_dir: str, on_log: Callable, api_key: str = ""):
        self.work_dir = work_dir
        self.on_log = on_log
        self.api_key = api_key or os.environ.get("STITCH_API_KEY", "")
        self.is_running = False

    async def generate_screens(self, app_description: str, screen_descriptions: list[str]) -> dict:
        """Generate UI screens using Stitch 2.0.

        Args:
            app_description: Overall app concept
            screen_descriptions: List of screen descriptions to generate

        Returns:
            Dict with project_id and list of generated screens
        """
        self.is_running = True
        await self.on_log("SYS", "Stitch 2.0 UI 디자인 생성 시작")

        try:
            # Generate design guide for FE Agent to follow
            result = await self._generate_via_prompt(app_description, screen_descriptions)
            await self.on_log("SYS", "Stitch 디자인 가이드 생성 완료")
            return result

        except Exception as e:
            await self.on_log("ERR", f"Stitch 오류: {str(e)}")
            return {}
        finally:
            self.is_running = False

    async def get_screen_code(self, project_id: str, screen_id: str) -> str:
        """Retrieve generated HTML/CSS code for a screen."""
        result = await self._run_stitch_tool("get_screen_code", {
            "projectId": project_id,
            "screenId": screen_id,
        })
        return result.get("code", "") if result else ""

    async def get_all_screens(self, project_id: str) -> list[dict]:
        """List all screens in a Stitch project."""
        result = await self._run_stitch_tool("list_screens", {
            "projectId": project_id,
        })
        return result.get("screens", []) if result else []

    async def _run_stitch_tool(self, tool_name: str, args: dict) -> Optional[dict]:
        """Run a Stitch MCP tool via CLI."""
        env = {**os.environ, "STITCH_API_KEY": self.api_key}

        cmd_args = ["npx", "@_davideast/stitch-mcp", "tool", tool_name]
        for key, value in args.items():
            cmd_args.extend([f"--{key}", str(value)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    await self.on_log("SYS", f"Stitch [{tool_name}]: {output[:200]}")
                    return {"raw": output}

            if stderr:
                err = stderr.decode("utf-8", errors="replace").strip()
                if err and "warn" not in err.lower():
                    await self.on_log("ERR", f"Stitch: {err[:200]}")

            return None

        except asyncio.TimeoutError:
            await self.on_log("ERR", f"Stitch tool '{tool_name}' 타임아웃")
            return None
        except Exception as e:
            await self.on_log("ERR", f"Stitch tool 오류: {str(e)}")
            return None

    async def _generate_via_prompt(self, app_description: str, screens: list[str]) -> dict:
        """Fallback: generate design descriptions for Claude to implement."""
        design_guide = {
            "app": app_description,
            "screens": [],
        }

        for screen in screens[:5]:
            design_guide["screens"].append({
                "name": screen,
                "description": f"Production-ready UI for: {screen}",
            })

        # Save design guide for FE Agent
        guide_path = os.path.join(self.work_dir, "stitch-design-guide.json")
        with open(guide_path, "w", encoding="utf-8") as f:
            json.dump(design_guide, f, ensure_ascii=False, indent=2)

        await self.on_log("SYS", "stitch-design-guide.json 저장됨")
        return design_guide

    async def save_designs_to_workspace(self, project_id: str, output_dir: str) -> list[str]:
        """Download all screen designs and save as HTML files."""
        saved = []
        screens = await self.get_all_screens(project_id)

        for screen in screens:
            screen_id = screen.get("id", screen.get("name", ""))
            if not screen_id:
                continue

            code = await self.get_screen_code(project_id, screen_id)
            if code:
                filename = f"stitch-{screen_id}.html"
                filepath = os.path.join(output_dir, filename)
                os.makedirs(output_dir, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)
                saved.append(filename)
                await self.on_log("SYS", f"디자인 저장: {filename}")

        return saved

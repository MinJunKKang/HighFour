# agents/clarify_agent.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

class ClarifyAgent:
    def __init__(
        self,
        client,
        prompt_path: str = "agents/prompts/clarify_or_redirect.prompt.md",
        model: str = "gpt-5.2",
    ):
        self.client = client
        self.model = model
        self.prompt_path = Path(prompt_path)
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def _build_prompt(self, user_input: str) -> str:
        return self._prompt_template.replace("{{user_input}}", user_input.strip())

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text.startswith("{"):
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                text = text[s:e+1]
        return json.loads(text)

    def run(self, user_input: str) -> Dict[str, Any]:
        prompt = self._build_prompt(user_input)

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = getattr(resp, "output_text", None) or str(resp)

        try:
            obj = self._parse_json(raw)
        except Exception:
            # fallback: 의료 모호로 처리
            return {
                "route": "clarify",
                "message": "정확한 도움을 드리기 위해 증상을 조금만 더 자세히 알려주세요.",
                "questions": ["언제부터 불편하셨나요?", "어느 부위가 어떻게 불편한가요?", "열/기침/통증 같은 동반 증상이 있나요?"],
            }

        route = obj.get("route")
        msg = obj.get("message")
        qs = obj.get("questions", [])

        if route not in ("clarify", "redirect"):
            route = "clarify"
        if not isinstance(qs, list):
            qs = []

        # 규약 강제
        if route == "redirect":
            qs = []

        return {"route": route, "message": str(msg or "").strip(), "questions": qs}
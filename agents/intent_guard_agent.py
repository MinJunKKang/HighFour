# agents/intent_guard_agent.py
from __future__ import annotations


import json
from pathlib import Path
from typing import Any, Dict

class IntentGuardAgent:
    """
    Intent Guard Agent
    ------------------
    역할:
    - 사용자의 입력이 의료 상담 의도가 있는지 1차 판별
    - medical / clarify / redirect 중 하나로 라우팅


    ❌ 하지 않는 것
    - 증상 추출
    - 진단 / 치료 / 병원 추천
    """

    def __init__(
            self,
            client,
            prompt_path: str = "agents/prompts/intent_guard.prompt.md",
            model: str = "gpt-5.2",
    ):
        self.client = client
        self.model = model
        self.prompt_path = Path(prompt_path)

        # Prompt 템플릿 -> 1회만 로드
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")

    def _build_prompt(self, user_input: str) -> str:
        """
        프롬프트 템플릿에 사용자 입력 주입
        """
        return self._prompt_template.replace("{{user_input}}", user_input.strip())
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """
        LLM 출력에서 JSON만 안전하게 추출
        """
        text = text.strip()


        if not text.startswith("{"):
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                text = text[s:e + 1]

        return json.loads(text)
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        단일 진입점
        반환 형식:
        {
        "intent": "medical" | "clarify" | "redirect",
        "message": str,
        "questions": list[str]
        }
        """

        prompt = self._build_prompt(user_input)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
        )

        raw_text = getattr(resp, "output_text", None) or str(resp)

        try:
            obj = self._parse_json(raw_text)
        except Exception:
            # 의도 판별 실패 시 redirect
            return {
                "intent": "redirect",
                "message": "이 서비스는 건강 관련 증상 상담을 위한 챗봇이에요. 도움이 필요하시면 증상을 알려주세요.",
                "questions": [],
            }

        intent = obj.get("intent")
        message = obj.get("message", "")
        questions = obj.get("questions", [])


        # ===== 규약 강제 =====
        if intent not in ("medical", "clarify", "redirect"):
            intent = "redirect"

        if intent == "medical":
            questions = []

        if intent == "redirect":
            questions = []

        if intent == "clarify" and not isinstance(questions, list):
            questions = []

        return {
            "intent": intent,
            "message": str(message).strip(),
            "questions": questions,
        }
    
# agents/safety_agent.py

from agents.prompts import load_prompt
import json


class SafetyAgent:
    """
    응급 여부 판단 + 안전 가드레일 에이전트
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = load_prompt("safety_notice.prompt.md")

    def run(self, symptoms: list, topk: list = None) -> dict:
        """
        반환 형식 (고정):
        {
            "is_emergency": bool,
            "reason": str
        }
        """

        messages = [
            {
                "role": "system",
                "content": self.prompt.render_system()
            },
            {
                "role": "user",
                "content": self.prompt.render_user({
                    "symptoms": symptoms,
                    "topk": topk
                })
            }
        ]

        raw = self.llm.chat(messages)

        # GPT는 JSON 문자열만 출력하도록 프롬프트에서 강제해야 함
        try:
            result = json.loads(raw["content"])
        except Exception:
            # LLM 출력이 깨졌을 경우의 안전 fallback
            result = {
                "is_emergency": False,
                "reason": "응급 여부를 명확히 판단할 수 없어 비응급으로 처리했습니다."
            }

        return result

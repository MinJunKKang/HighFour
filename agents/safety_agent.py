# agents/safety_agent.py

from agents.prompts.loader import load_prompt
import json


class SafetyAgent:
    """
    응급 여부 판단 + 안전 가드레일 에이전트
    """

    def __init__(self, llm):
        self.llm = llm
        # ❗ prompt는 이제 그냥 문자열
        self.system_prompt = load_prompt("safety_notice.prompt.md")

    def run(self, symptoms: list, topk: list = None) -> dict:
        """
        반환 형식 (고정):
        {
            "is_emergency": bool,
            "reason": str
        }
        """

        user_prompt = f"""
증상 목록:
{symptoms}

질병 Top-K 예측 결과:
{topk}

위 정보를 참고하여 응급 상황 여부를 판단하라.
반드시 JSON 형식으로만 응답하라.
"""

        messages = [
            {
                "role": "system",
                "content": self.system_prompt   # ✅ render_system ❌
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        raw = self.llm.responses.create(
            model="gpt-5.2",
            input=messages
        )

        try:
            result = json.loads(raw.output_text)
        except Exception:
            result = {
                "is_emergency": False,
                "reason": "응급 여부를 명확히 판단할 수 없어 비응급으로 처리했습니다."
            }


        return result

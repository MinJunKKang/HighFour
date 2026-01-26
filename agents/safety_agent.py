# agents/safety_agent.py

from agents.prompts.loader import load_prompt
import json


class SafetyAgent:
    """
    응급 여부 판단 + 안전 가드레일 에이전트
    """

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = load_prompt("safety_notice.prompt.md")

    def run(self, symptoms: list, topk: list = None) -> dict:
        """
        반환 형식 (고정):
        {
            "is_emergency": bool,
            "total_score": int,
            "technical_reason": str,
            "user_reason": str
        }
        """
        user_prompt = f"""
증상 목록:
{symptoms}

질병 Top-K 예측 결과:
{topk}

위 정보를 참고하여 응급 상황 여부를 판단하라.
응급 여부, 점수 합산 근거, 사용자용 안내를 포함하여 JSON으로 반환하라.
반드시 다음 키를 포함할 것: is_emergency, total_score, technical_reason, user_reason
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw = self.llm.responses.create(
            model="gpt-5.2",
            input=messages
        )

        try:
            result = json.loads(raw.output_text)
        except Exception:
            # 파싱 실패 시 안전하게 기본값 반환
            result = {
                "is_emergency": False,
                "total_score": 0,
                "technical_reason": "GPT 응답 파싱 실패",
                "user_reason": "응급 여부를 명확히 판단할 수 없어 비응급으로 처리했습니다."
            }

        # key 누락 방지
        for key in ["is_emergency", "total_score", "technical_reason", "user_reason"]:
            if key not in result:
                result[key] = None if key != "total_score" else 0

        return result
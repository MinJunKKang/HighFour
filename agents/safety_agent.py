# agents/safety_agent.py

from agents.prompts.loader import load_prompt
import json
import re


class SafetyAgent:
    """
    응급 여부 판단 + 안전 가드레일 에이전트
    """

    # 정신건강 키워드(부분일치) - 최소/실용 세트
    MENTAL_PATTERNS = [
        # EN
        r"\bdepress", r"\banx", r"\bpanic", r"\bptsd\b", r"\bocd\b",
        r"\bbipolar\b", r"\bschizo", r"\bpsych", r"\bsuicid",
        r"\bself[- ]?harm\b", r"\binsomnia\b", r"\bstress\b", r"\badhd\b", r"\btrauma\b",
        # KO
        r"우울", r"불안", r"공황", r"사회불안", r"강박", r"불면|수면장애", r"트라우마",
        r"자해", r"자살", r"환청|환각|망상", r"조울|양극성",
    ]

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = load_prompt("safety_notice.prompt.md")
        self._mental_rx = [re.compile(p, re.IGNORECASE) for p in self.MENTAL_PATTERNS]

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

        # ✅ 요구사항: 정신질환 키워드 포함되면 응급 판단은 무조건 False
        sym_text = " ".join(str(s) for s in (symptoms or []))
        if any(rx.search(sym_text) for rx in self._mental_rx):
            result["is_emergency"] = False

        return result

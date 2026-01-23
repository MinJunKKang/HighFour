# 2026-01-23 송진우
"""
Explain Agent : 비진단적 행동 가이드 에이전트

역할:
- ML 모델이 산출한 Top-K 질병 후보를 비진단적 · 사용자 친화적 설명으로 변환

판단 X 사용자에게 설명 및 가이드 라인 제시
"""

from agents.prompts import load_prompt
from tools.schemas import ExplainAgentInput, ExplainAgentOutput


class ExplainAgent:

    def __init__(self, llm):
        """
        llm : main에서 생성한 GPT 객체 받아옴
        """
        self.llm = llm
        self.prompt = load_prompt("explain_topk.prompt.md") # 사전에 정의된 prompt를 로드

    def run(self, input_data: ExplainAgentInput) -> ExplainAgentOutput:
        """
        응답 생성

        input_data
        - 증상 정규화 완료
        - Top-K 선별 완료
        - 응급 여부 판단 완료
        """

        messages = [
            {
                "role": "system",
                "content": self.prompt.render_system()
            },
            {
                "role": "user",
                "content": self.prompt.render_user({
                    "symptoms": input_data.normalized_symptoms,
                    "topk": input_data.topk,
                    "emergency": input_data.emergency
                })
            }
        ]

        response = self.llm.chat(messages)

        # JSON 출력 강제 (schemas.py에서 검증)
        return ExplainAgentOutput.parse_raw(response)
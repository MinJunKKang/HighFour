from agents.prompts.loader import load_prompt
import json


class ExplainAgent:
    """
    질병 Top-K 결과를 사용자 친화적으로 설명하는 에이전트
    """

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = load_prompt("explain_topk.prompt.md")

    def run(self, input_data: dict) -> str:
        """
        input_data 예시:
        {
            "symptoms": [...],
            "topk": [...],
            "disease_info": {...}
        }
        """

        user_prompt = f"""
사용자 증상:
{input_data.get("symptoms")}

예측된 질병 Top-K:
{input_data.get("topk")}

질병 관련 정보:
{input_data.get("disease_info")}
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.llm.responses.create(
            model="gpt-5.2",
            input=messages
        )

        return resp.output_text

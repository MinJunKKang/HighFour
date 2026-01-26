# agents/explain_agent.py

from agents.prompts.loader import load_prompt


class ExplainAgent:
    """
    Top-K 질병 후보를 사용자 친화적으로 설명
    - 비진단
    - 점수/확률 언급 금지
    """

    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = load_prompt("explain_topk.prompt.md")

    def run(self, input_data: dict) -> str:
        """
        input_data:
        {
            "symptoms": list[str],
            "topk": list[str]
        }
        """

        user_prompt = f"""
사용자 증상 (정규화된 리스트):
{input_data["symptoms"]}

의심되는 질환 후보 (순서만 의미 있음):
{input_data["topk"]}

위 정보를 바탕으로,
- 의료 진단이 아님을 분명히 밝히고
- 각 질환이 어떤 경우에 고려될 수 있는지
- 증상과의 일반적인 연관성만 설명하세요
- 점수, 확률, 순위, 정확도 같은 표현은 절대 사용하지 마세요
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.llm.responses.create(
            model="gpt-5.2",
            input=messages,
        )

        return resp.output_text

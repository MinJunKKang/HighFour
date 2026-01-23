# 2026-01-23 송진우
"""
Explain Agent : 비진단적 행동 가이드 에이전트

역할:
- ML 모델이 산출한 Top-K 질병 후보를
  사용자 친화적 · 비진단적 설명으로 변환
"""

from typing import List, Dict, Any
from agents.prompts import load_prompt

class ExplainAgent:

    def __init__(self, llm):
        self.llm = llm
        self.prompt = load_prompt("explain_topk.prompt.md")

    def run(
        self,
        symptoms: List[str],
        topk: List[Dict[str, Any]],
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": self.prompt.render_system()
            },
            {
                "role": "user",
                "content": self.prompt.render_user({
                    "symptoms": symptoms,
                    "topk": topk,
                })
            }
        ]

        response = self.llm.chat(messages)

        # 🔴 이 줄이 핵심
        return response["content"]


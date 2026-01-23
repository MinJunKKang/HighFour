# 응급 여부 판단 보조 + 안전 가드레일 적용(금지 발화 방지)

from agents.prompts import load_prompt
from tools.schemas import SafetyCheckResult

class SafetyAgent:
    def __init__(self, llm):
        self.llm = llm
        # 안전 지침이 담긴 프롬프트 로드
        self.prompt = load_prompt("safety_notice.prompt.md")

    def check(self, symptoms: list) -> SafetyCheckResult:
        """
        추출된 증상 리스트를 분석하여 응급 여부 및 입력의 적절성 판단
        """
        messages = [
            {
                "role": "system",
                "content": self.prompt.render_system()
            },
            {
                "role": "user",
                "content": self.prompt.render_user({
                    "symptoms": symptoms
                })
            }
        ]
        
        response = self.llm.chat(messages)
        
        return SafetyCheckResult.parse_raw(response)
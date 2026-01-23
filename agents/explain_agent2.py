import os
from openai import OpenAI
# __init__.py에 정의된 경로를 가져옴
from . import EXPLAIN_PROMPT_PATH 

class ResponseAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        # __init__.py에서 가져온 경로를 그대로 할당
        self.prompt_path = EXPLAIN_PROMPT_PATH

    def _load_prompt(self):
        """정의된 경로에서 프롬프트 파일을 읽어옵니다."""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def generate_explanations(self, top_k_results):
        # 1. 질병 후보군 리스트 가공 
        disease_info_str = "\n".join([
            f"- {res['disease']} (분석 일치도: {res['prob']*100:.1f}%)" 
            for res in top_k_results
        ])

        # 2. 템플릿 로드 및 치환
        template = self._load_prompt()
        final_prompt = template.replace("{{disease_info}}", disease_info_str)

        # 3. LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 건강 데이터 분석 결과를 안전하게 설명하는 AI 보조원이야."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
# agents/symptom_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class SymptomExtractResult:
    """
    최종 결과를 반환하는 DTO
    - symptoms: canonical(영문 증상명 리스트)
    - unmapped_phrase: 증상인거 같은데? vocab에 없는 사용자의 표현
    """


class SymptomAgent:
    """
    - 사용자 자연어(다국어) 입력을 받아
    - symptom_vocab.json에 정의된 377개 canonical 중에서 증상을 골라내어
    - JSON 형태로 반환한다.
    """

    def __init__(
            self,
            client,                                                             # main.py에서 생성한 OpenAI client를 주입
            prompt_path: str = "agents/prompts/symptom_extract.prompt.md",      # Prompt 템플릿 경로
            vocab_path: str = "ml/artifacts/symptom_vocab.json",                # Symptom vocab 파일 경로
            model: str = "gpt-5.2",                                             # 사용할 Model명
    ):
        self.client = client
        self.model = model

        # 파일 경로를 Path 객체로 관리
        self.prompt_path = Path(prompt_path)
        self.vocab_path = Path(vocab_path)

        # Prompt 템플릿 load (한 번만)
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")

        # Symptom_vocab.json에서 canonical 리스트 로드
        self._allowed_symptoms = self._load_allowed_symptoms(self.vocab_path)

        # allowed_symptoms를 JSON 배열 문자열로 미리 만들어두기
        # (매 요청마다 dumps 하는 것이 아닌, 고정된 문자열을 주입하면 안정적이기 때문)
        self._allowed_symptoms_json = json.dumps(self._allowed_symptoms, ensure_ascii=False)

    def _load_allowed_symptoms(self, vocab_path: Path) -> list[str]:
        """"
        symptom_vocab.json을 읽어서 canonical 리스트만 뽑아내기.
        - {"version": "...", "symptoms": [{"canonical": "...", "ko": "...", "aliases": [...]}, ...]}
        여기서 canonical만 추출
        """

        return
    
    def _build_prompt(self, user_input: str) -> str:
        """
        템플릿 프롬프트에 사용자 입력과 allowed symptom 리스트를 주입하기
        - {{user_input}}
        - {{allowed_symptoms_json}}
        """

        return
    
    def _parse_json(self, text:str) -> dict[str, Any]:
        """
        모델의 출력은 JSON 형태로 오겠지만, 가끔 추가적인 문장이 섞일 수 있기 때문에 파싱 로직을 추가한다.
        """

        return
    
    def vaidate(self, obj: dict[str, Any]) -> SymptomExtractResult:
        """
        모델이 준 JSON을 검증
        - symptoms는 allowed vocab 안에 있는 canonical만 남기기
        - 중복 제거하기
        """

        return
    
    def extract(self, user_input: str) -> SymptomExtractResult:
        """
        외부에서 호출할 메인 함수
        - 사용자 입력 -> 프롬프트 생성 -> GPT 호출 -> JSON을 파싱 및 검증 -> 결과를 반환
        """

        return

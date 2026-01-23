# agents/symptom_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class SymptomExtractResult:
    # ML에 넘길 canonical 영문 증상명 리스트(377개 vocab 중에서만)
    symptoms: list[str]


class SymptomAgent:
    """
    - 사용자 자연어(다국어) 입력을 받아
    - symptom_vocab.json에 정의된 377개 canonical 중에서 증상을 골라내어
    - {"symptoms":[...]} 형태로 반환
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
        # 매번 읽으면 속도가 느려짐 -> 캐싱
        self._allowed_set = set(self._allowed_symptoms)

        # allowed_symptoms를 JSON 배열 문자열로 미리 만들어두기
        # (매 요청마다 dumps 하는 것이 아닌, 고정된 문자열을 주입하면 안정적이기 때문)
        self._allowed_symptoms_json = json.dumps(self._allowed_symptoms, ensure_ascii=False)

    def _load_allowed_symptoms(self, vocab_path: Path) -> list[str]:
        """
        symptom_vocab.json을 읽어서 canonical 리스트만 뽑아내기.
        - {"version": "...", "symptoms": [{"canonical": "...", "ko": "...", "aliases": [...]}, ...]}
        여기서 canonical만 추출
        """

        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        symptoms = data.get("symptoms", [])

        canonicals: list[str] = []
        for item in symptoms:
            if not isinstance(item, dict):
                continue
            c = (item.get("canonical") or "").strip()
            if c:
                canonicals.append(c)

        # 중복을 제거하기 위한 seen 집합
        seen: set[str] = set()
        uniq: list[str] = []
        for c in canonicals:
            if c not in seen:
                uniq.append(c)
                seen.add(c)

        if not uniq:
            raise ValueError("symptom_vocab.json loaded but no canonical symptoms found.")

        return uniq
    
    def _build_prompt(self, user_input: str) -> str:
        """
        템플릿 프롬프트에 사용자 입력과 allowed symptom 리스트를 주입하기
        - {{user_input}}
        - {{allowed_symptoms_json}}
        """
        prompt = self._prompt_template
        prompt = prompt.replace("{{user_input}}", user_input.strip())
        prompt = prompt.replace("{{allowed_symptoms_json}}", self._allowed_symptoms_json)

        return prompt
    
    def _parse_json(self, text:str) -> dict[str, Any]:
        """
        모델의 출력은 JSON 형태로 오겠지만, 가끔 추가적인 문장이 섞일 수 있기 때문에 파싱 로직을 추가

        예시 )
        물론입니다! 아래는 추출 결과입니다.
        {"symptoms":["cough","fever"]}
        """

        text = text.strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        return json.loads(text)
    
    def _validate(self, obj: dict[str, Any]) -> SymptomExtractResult:
        """
        모델이 준 JSON을 검증
        - symptoms는 allowed vocab 안에 있는 canonical만 남기기
        - 중복 제거하기
        """
        symptoms = obj.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = []

        cleaned: list[str] = []
        seen: set[str] = set()

        for s in symptoms:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if s in self._allowed_set and s not in seen:
                cleaned.append(s)
                seen.add(s)

        return SymptomExtractResult(symptoms=cleaned)
    
    def extract(self, user_input: str) -> SymptomExtractResult:
        """
        외부에서 호출할 메인 함수
        - 사용자 입력 -> 프롬프트 생성 -> GPT 호출 -> JSON을 파싱 및 검증 -> 결과를 반환
        """

        prompt = self._build_prompt(user_input)

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        raw_text = getattr(resp, "output_text", None) or str(resp)

        # 파싱 실패하면 빈 리스트 반환(상위 UI에서 “더 구체화” 유도)
        try:
            obj = self._parse_json(raw_text)
        except Exception:
            return SymptomExtractResult(symptoms=[])

        return self._validate(obj)
    
    def run(self, user_input: str) -> list[str]:
        """
        Orchestrator에서 사용하는 단일 진입점
        - 내부 구현을 숨길 수 있는 장점
        """
        return self.extract(user_input).symptoms

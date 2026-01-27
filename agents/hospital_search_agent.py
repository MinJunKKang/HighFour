from typing import List, Dict, Optional, Any, Tuple
import json
import re


class HospitalSearchAgent:
    """
    - symptoms 토큰이 다양해도(영/한, 문장형, 복합형) 정신과 신호를 robust하게 감지
    - 정신과면 '병원' 대신 '정신건강의학과의원/정신과의원/멘탈클리닉' 위주로 검색
    - emergency 판단은 hospital_search_agent에서 하지 않음
      (입력 emergency 플래그는 '정신 신호가 없을 때만' 반영)
    - 출력 스키마는 Streamlit 렌더링에 맞춰 고정
    - OpenAI API only + OpenAI 내부 web_search 허용
    """

    # 정신건강 신호 패턴(부분일치)
    # - 토큰이 "depressive or psychotic symptoms" 같이 길어도 잡히도록 stem/substring 위주
    MENTAL_PATTERNS = [
        # EN
        r"\bdepress",         # depression, depressive...
        r"\banx",             # anxiety, anxious...
        r"\bpanic",           # panic
        r"\bptsd\b",
        r"\bocd\b",
        r"\bbipolar\b",
        r"\bschizo",          # schizophrenia...
        r"\bpsych",           # psychotic, psychiatry...
        r"\bsuicid",          # suicidal...
        r"\bself[- ]?harm\b",
        r"\binsomnia\b",
        r"\bstress\b",
        r"\bphobia\b",
        r"\badhd\b",
        r"\btrauma\b",

        # KO
        r"우울",
        r"불안",
        r"공황",
        r"대인",
        r"사회불안",
        r"강박",
        r"수면장애|불면",
        r"트라우마",
        r"자해",
        r"자살",
        r"환청|환각|망상",
        r"조울|양극성",
    ]

    # 정신과 “검색 키워드” (병원보다 의원/클리닉이 잘 잡힘)
    MENTAL_FACILITY_QUERY = "정신건강의학과의원 정신과의원 멘탈클리닉 마음클리닉 상담 클리닉"

    # 응급(입력 emergency=True일 때만) 검색 키워드
    EMERGENCY_FACILITY_QUERY = "응급실 24시간 응급의료센터 권역응급의료센터"

    # 일반 검색 키워드
    GENERAL_FACILITY_QUERY = "병원 의원"

    def __init__(self, client):
        """
        client는 OpenAI SDK 클라이언트(Responses API)를 가정한다.
        - self.client.responses.create(...) 호출 가능해야 함
        """
        self.client = client
        self._mental_regex = [re.compile(p, re.IGNORECASE) for p in self.MENTAL_PATTERNS]

    def run(
        self,
        symptoms: List[str],
        topk: List[str],  # 시그니처 유지(참고용으로만 사용)
        location: Optional[str] = None,
        emergency: bool = False,
    ) -> Dict[str, Any]:
        """
        [반환 스키마]
        {
          "status": "ok" | "partial" | "error",
          "message": str,
          "hospitals": [
            {"name","address","phone","latitude","longitude","department"}, ...
          ],
          "raw": str
        }
        """

        # 위치가 없으면 검색이 불가능하므로 에러 처리
        if not location or not str(location).strip():
            return {
                "status": "error",
                "message": "위치 정보가 없습니다. 예: '서울시 강남구'처럼 입력해주세요.",
                "hospitals": [],
                "raw": "",
            }

        # 1) 토큰 정규화 + 정신/비정신 분리
        mental_syms, other_syms = self._split_symptoms(symptoms)
        is_mental = bool(mental_syms)

        # 2) 라우팅 정책(핵심 수정)
        #    - 정신 신호가 하나라도 있으면 emergency=True여도 정신과가 최우선
        #    - emergency는 정신 신호가 없을 때만 반영
        if is_mental:
            department = "정신건강의학과"
            facility_query = self.MENTAL_FACILITY_QUERY
            note = "정신과 신호 토큰 감지(최우선) → 정신건강의학과(의원/클리닉) 우선"
        elif emergency:
            department = "응급의학과"
            facility_query = self.EMERGENCY_FACILITY_QUERY
            note = "emergency=True(외부 판단) → 응급실 중심 검색"
        else:
            department = ""
            facility_query = self.GENERAL_FACILITY_QUERY
            note = "증상 토큰 기반 일반 의료기관 검색(관련 진료과 표기된 곳 우선)"

        # 3) 검색에 넣을 증상 키워드 구성(핵심 수정)
        #    - 정신 모드면 '정신 토큰만' 사용하여 응급/흉통 등 other 토큰이
        #      모델 출력을 응급실로 끌고 가는 것을 차단한다.
        if is_mental:
            ordered_tokens = mental_syms[:10]
        else:
            ordered_tokens = (mental_syms + other_syms)[:10]

        symptom_text = self._sanitize(", ".join(ordered_tokens)) if ordered_tokens else ""

        # (참고) topk도 소량 힌트로만 포함(결정 로직에는 영향 없음)
        topk_hint = self._sanitize(", ".join([str(x) for x in (topk or [])[:3]]))

        # 4) 정신 모드에서는 프롬프트에 "응급실/센터 금지"를 강제한다(추가 안전장치)
        mental_only_constraint = ""
        if is_mental:
            mental_only_constraint = (
                "6) 정신건강 관련 신호가 포함되어 있으므로 반드시 정신건강의학과의원/정신과의원/클리닉만 포함해\n"
                "7) 응급실/응급의료센터/권역응급센터/종합병원 응급센터 등은 포함하지 마\n"
                "8) 응급 여부, 심장질환 등 의학적 추정/진단 문구는 절대 쓰지 마(오직 기관 정보만 JSON으로)\n"
            )

        # 5) 프롬프트 (JSON만 출력 강제 + 정신과는 의원/클리닉 우선 명시)
        query = (
            f"{self._sanitize(location)} 근처에서 아래 증상/상황에 맞는 의료기관 3곳을 찾아줘.\n"
            f"- 증상 토큰(우선순위 반영): {symptom_text or '-'}\n"
            f"- (참고) 예측 후보: {topk_hint or '-'}\n"
            f"- 우선 검색 키워드: {facility_query}\n"
            f"- 참고: {note}\n\n"
            "조건:\n"
            "1) 반드시 실제 존재하는 의료기관만(가능하면 지도/홈페이지/전화번호 근거)\n"
            "2) 같은 곳 중복 금지\n"
            "3) address는 꼭 포함, phone 없으면 '-'로\n"
            "4) latitude/longitude는 가능하면 채우고 모르면 null\n"
            "5) 정신건강의학과 케이스면 '병원'보다 '의원/클리닉'을 우선함\n"
            f"{mental_only_constraint}\n"
            "아래 JSON 한 덩어리만 출력(다른 텍스트 금지):\n"
            "{\n"
            '  "hospitals": [\n'
            "    {\n"
            '      "name": "",\n'
            '      "address": "",\n'
            '      "phone": "-",\n'
            '      "latitude": null,\n'
            '      "longitude": null,\n'
            f'      "department": "{department}"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        # OpenAI Responses API 호출 + OpenAI 내부 web_search 사용(허용 범위 내)
        response = self.client.responses.create(
            model="gpt-5.2",
            tools=[{"type": "web_search"}],
            input=query,
        )

        parsed = self._parse(response.output_text)
        return self._postprocess(
            parsed,
            fallback_department=department or ("정신건강의학과" if is_mental else ""),
        )

    # =========================================================
    # Token handling
    # =========================================================
    def _split_symptoms(self, symptoms: List[str]) -> Tuple[List[str], List[str]]:
        """
        symptoms 토큰이 다양하게 들어와도:
        - 소문자/공백 정리
        - 불필요한 기호 제거
        - 정신과 신호(부분일치)면 mental로 분리
        """
        mental: List[str] = []
        other: List[str] = []

        for s in symptoms or []:
            raw = str(s).strip()
            if not raw:
                continue

            norm = self._normalize_token(raw)

            # 너무 짧은 토큰은 의미 없으니 제외(예: '-', 'a')
            if len(norm) < 2:
                continue

            if self._is_mental_token(norm):
                mental.append(norm)
            else:
                other.append(norm)

        # 중복 제거(순서 유지)
        return self._dedupe_keep_order(mental), self._dedupe_keep_order(other)

    def _is_mental_token(self, token: str) -> bool:
        """
        token에 정신과 패턴이 하나라도 매치되면 True.
        """
        for rx in self._mental_regex:
            if rx.search(token):
                return True
        return False

    def _normalize_token(self, token: str) -> str:
        """
        토큰 정규화:
        - 소문자화
        - 구분자(/,|)를 공백으로 치환
        - 중복 공백 제거
        - 프롬프트/JSON을 깨뜨릴 수 있는 기호 제거
        """
        t = token.lower()
        t = t.replace("_", " ")
        t = re.sub(r"[/|]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = t.replace("{", "").replace("}", "").replace("`", "").replace('"', "")
        return t

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        """
        중복 제거(입력 순서 유지).
        """
        seen = set()
        out: List[str] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def _sanitize(self, s: str) -> str:
        """
        프롬프트에 넣기 전 문자열 정리.
        """
        s = str(s).replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = s.replace("{", "").replace("}", "").replace("`", "")
        return re.sub(r"\s+", " ", s).strip()

    # =========================================================
    # JSON parsing (robust)
    # =========================================================
    def _parse(self, text: str) -> Dict[str, Any]:
        """
        모델 출력에서 JSON 객체를 추출해 dict로 반환한다.
        """
        if not text:
            return {"status": "partial", "message": "빈 응답", "hospitals": [], "raw": ""}

        # 코드펜스 제거(있으면)
        cleaned = re.sub(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            r"\1",
            text,
            flags=re.IGNORECASE,
        ).strip()

        obj = self._extract_first_json_object(cleaned)
        if isinstance(obj, dict):
            obj.setdefault("hospitals", [])
            obj.setdefault("raw", "")
            return obj

        return {"status": "partial", "message": "JSON 파싱 실패", "hospitals": [], "raw": text}

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """
        문자열에서 첫 번째 JSON 객체를 찾아 파싱한다.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    # =========================================================
    # Postprocess for Streamlit schema + dedupe
    # =========================================================
    def _postprocess(self, parsed: Dict[str, Any], fallback_department: str) -> Dict[str, Any]:
        """
        Streamlit 출력 스키마로 정리:
        - name/address 필수
        - 중복 제거
        - 위/경도 float 변환
        - 최대 3개
        """
        hospitals = parsed.get("hospitals", [])
        raw = parsed.get("raw", "")

        out: List[Dict[str, Any]] = []
        seen = set()

        if not isinstance(hospitals, list):
            hospitals = []

        for h in hospitals:
            if not isinstance(h, dict):
                continue

            name = str(h.get("name") or "").strip()
            address = str(h.get("address") or "").strip()
            if not name or not address:
                continue

            key = (self._norm_key(name), self._norm_key(address))
            if key in seen:
                continue
            seen.add(key)

            out.append({
                "name": name,
                "address": address,
                "phone": str(h.get("phone") or "-").strip() or "-",
                "latitude": self._to_float_or_none(h.get("latitude")),
                "longitude": self._to_float_or_none(h.get("longitude")),
                "department": str(h.get("department") or fallback_department).strip(),
            })

        status = parsed.get("status") or ("ok" if out else "partial")
        message = parsed.get("message")
        if message is None:
            message = "" if out else "병원 정보를 충분히 찾지 못했습니다. 위치를 더 구체적으로 입력해 보세요."

        return {
            "status": status,
            "message": message,
            "hospitals": out[:3],
            "raw": raw if status != "ok" else "",
        }

    def _to_float_or_none(self, v: Any) -> Optional[float]:
        """
        위/경도 타입 정리.
        """
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _norm_key(self, s: str) -> str:
        """
        중복 판정을 위한 정규화 키 생성.
        """
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s가-힣]", "", s)
        return s.strip()

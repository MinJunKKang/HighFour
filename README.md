# 🩺 HighFour

**자연어 기반 증상 입력 → 질병 후보 예측 → 비진단적 행동 가이드 제공 시스템**

> 사용자의 자연어 증상 입력을 바탕으로
> **증상 추출 → 질병 후보 예측 → 응급 여부 판단 → 행동 가이드 제공**까지
> 하나의 파이프라인으로 처리하는 **비진단 의료 지원 챗봇 프로젝트**

---

## 📌 프로젝트 개요

HighFour는 “진단하지 않는 의료 지원”을 목표로 합니다.

* ❌ 병명 확정 / 치료 지시 / 약물 추천 ❌
* ✅ 증상 이해 + 위험도 판단 + 다음 행동 가이드 ✅

사용자는 일상적인 언어로 증상을 입력하고,
시스템은 **구조화된 증상 → 질병 후보 → 안전한 안내** 흐름으로 응답합니다.

---

## 프로젝트 목표

### 목표 1

**사용자의 자연어 기반 대화에서 증상 추출**

* 자유로운 문장 입력 지원
* LLM 기반 표준 증상 리스트로 정규화

### 목표 2

**추출된 증상 기반 상위 질병 후보 예측**

* ML 모델을 통한 Top-K 질병 라벨 예측
* 점수는 내부 판단용으로만 사용

### 목표 3

**비진단적 행동 가이드 제공**

* 응급 여부 판단
* 병원 방문 필요 시 정보 제공
* 사용자가 스스로 다음 행동을 결정할 수 있도록 지원

---

## 전체 시스템 흐름

```
User
 ↓
Streamlit UI
 ↓
Orchestrator (Pipeline Controller)
 ↓
Intent Guard (의료 의도 판별)
 ├─ 의료 아님 → Redirect
 ├─ 의료지만 모호 → Clarify
 └─ 의료 의도 확정
        ↓
   Symptom Agent (증상 추출)
        ↓
   ML Predictor (질병 후보 Top-K)
        ↓
   Safety Agent (응급 여부 판단)
        ├─ Emergency → Hospital Search
        └─ Non-Emergency → Explain Agent
```

---

## 🧩 핵심 구성 요소

### 🧠 Orchestrator

* 전체 파이프라인을 **상태 없이(stateless)** 제어
* 각 Agent 간 흐름 및 분기만 담당

### 🛡 Intent Guard Agent

* 입력이 **의료 관련인지 / 모호한지 / 무관한지** 1차 판별
* 노래 가사, 농담, 비유적 표현 필터링

### 🩻 Symptom Agent

* 자연어 문장에서 **표준화된 증상 리스트 추출**
* ML/안전 판단의 단일 입력 소스

### 🤖 ML Predictor

* 증상 리스트 기반 **질병 후보 Top-K 예측**
* 사용 모델:

  * Logistic Regression (L1 / L2)
  * Bernoulli Naive Bayes
  * XGBoost

### 🚨 Safety Agent

* 증상 + 질병 후보를 기반으로 **응급 여부 판단**
* 응급 상황일 경우 즉시 병원 정보 분기

### 🏥 Hospital Search Agent

* 위치 기반 병원 정보 제공
* 응급 / 비응급 분리 처리

### 📖 Explain Agent

* **진단이 아닌 설명 중심**
* 질병 후보에 대한 일반적 정보 + 행동 가이드 제공

---

## 테스트 및 검증 전략

* Intent Guard → Symptom Agent → ML → Safety → Explain
* 시나리오 기반 테스트

  * 의료 무관 입력
  * 모호한 증상 입력
  * 증상은 있으나 추출 실패
  * 응급 상황
  * 일반 비응급 상황

---

## 🛠 사용 기술 스택

| 구분              | 기술                    |
| --------------- | --------------------- |
| Language        | Python                |
| UI              | Streamlit             |
| LLM API         | OpenAI                |
| ML              | Scikit-learn, XGBoost |
| Data            | Kaggle                |
| Data Handling   | pandas, NumPy         |
| Version Control | GitHub                |
| Env             | Google Colab / Local  |

---

## 📂 프로젝트 구조

```
agents/
 ├─ orchestrator.py
 ├─ intent_guard_agent.py
 ├─ symptom_agent.py
 ├─ safety_agent.py
 ├─ explain_agent.py
 ├─ hospital_search_agent.py
 └─ prompts/

ml/
 ├─ train/
 └─ artifacts/

tools/
 └─ ml_predict_tool.py

ui/
 └─ streamlit_app.py

app/
 ├─ main.py
 └─ config.py
```

---

## ⚠️ 주의 사항

* 본 프로젝트는 **의료 진단 시스템이 아닙니다**
* 모든 출력은 참고용 정보이며,
  **실제 의학적 판단은 의료 전문가와 상담해야 합니다**

---

## 👥 Team HighFour

> “AI로 진단하지 말고, 판단을 돕자.”

| 이름      | 담당 역할 (요약)                                                                           |
| ------- | ------------------------------------------------------------------------------------ |
| **강민준** | Intent Guard·Symptom Agent 설계, Orchestrator 리팩토링 및 파이프라인 제어, Streamlit Chat UI 구조 개선 |
| **김경희** | XGBoost·CatBoost·RandomForest 모델 구현, ML 평가 코드 및 성능 분석 리포트 작성, 아키텍처 흐름도               |
| **김민주** | Safety Agent(가중치 기반 응급 판정) 구현, LLM 응급/안전 가드레일 설계, ML 보조 학습 및 성능 검증                   |
| **김세빈** | Hospital Search Agent 구현(Web Search 연동), 정신과/응급 분기 로직 보완, 병원 검색 워크플로우 설계             |
| **송진우** | Explain Agent 구현(비진단적 가이드), Orchestrator 구현 및 규칙 기반 분기, 통합 테스트·발표 자료 제작              |
| **윤성원** | Streamlit UI 설계 및 구현, CatBoost 모델 학습·연동, 사용자 흐름 중심 UI/UX 개선                          |
| **전민주** | ML 알고리즘 조사, XGBoost·RandomForest 성능 비교 실험, 데이터 전처리 및 지표 기반 모델 검증                     |

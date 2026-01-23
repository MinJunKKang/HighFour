# app/main.py

from openai import OpenAI
from app.config import OPENAI_API_KEY

from agents import (
    SymptomAgent,
    SafetyAgent,
    ExplainAgent,
    HospitalSearchAgent,
)
from agents.orchestrator import Orchestrator
from tools import MLPredictTool


def create_orchestrator() -> Orchestrator:

    # 1️⃣ GPT-5.2 Client 단일 생성
    llm_client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    # 2️⃣ Agents (모두 동일한 llm 공유)
    symptom_agent = SymptomAgent(llm_client)
    safety_agent = SafetyAgent(llm_client)
    explain_agent = ExplainAgent(llm_client)
    hospital_search_agent = HospitalSearchAgent(llm_client)

    # 3️⃣ ML Tool
    ml_predict_tool = MLPredictTool()

    # 4️⃣ Orchestrator
    orchestrator = Orchestrator(
        symptom_agent=symptom_agent,
        safety_agent=safety_agent,
        explain_agent=explain_agent,
        hospital_search_agent=hospital_search_agent,
        ml_predict_tool=ml_predict_tool,
    )

    return orchestrator

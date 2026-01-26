# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
# Streamlit의 경우 ui/ 폴더에서 진행 -> CWD가 ui/가 될 수도 있음
# dotenv_path=ENV_PATH로 위와 같은 문제를 해결
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# .env가 있으면 로드, 없어도 환경변수로 동작 가능
# overried=False란? - OS 환경변수로 설정된 값이 있으면 그걸 우선시
# 배포 환경(EC2, EB, Docker 등)에서는 보통 시스템 환경 변수로 키를 주입하기 때문 (경우에 따라서는 .env를 환경변수로 사용하도록 주입할 수 있음)
load_dotenv(dotenv_path=ENV_PATH, override=False)

# OpenAI Key
OPENAI_API_KEY = OPENAI_API_KEY="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"

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Please set it in .env (PROJECT_ROOT/.env) or environment variables."
    )
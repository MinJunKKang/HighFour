"""
tools/ml_predict_tools2.py

요구사항:
- 다른 파일(a.py 등)에서 import 해서 함수 호출
- 모델 경로/아티팩트 경로 등은 이 모듈 내부에 하드코딩
- 입력: 증상 리스트(list[str])  (피처명 기준)
- 출력: Top-5 질병 후보 리스트(list[str])

추가:
- 최초 1회만 모델/아티팩트 로드(캐시) -> 여러 번 호출해도 빠름
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Optional
import json

import numpy as np
import xgboost as xgb

from dataclasses import dataclass
from typing import Dict, Any

# =========================
# 1) 하드코딩 설정(여기만 맞추면 됨)
# =========================
# a.py를 실행하는 "현재 작업 디렉토리" 기준 상대경로가 꼬일 수 있어서,
# 가능하면 절대경로로 바꾸거나, 프로젝트 루트 기준으로 계산하는 방식을 추천.
# MODEL_PATH = Path("ml_project_skeleton_v2/ml/artifacts/xgb_model.json")
# ARTIFACTS_DIR = Path("ml_project_skeleton_v2/ml/artifacts")

# 이 파일(tools/ml_predict_tools3.py) 기준으로 레포 루트 계산
REPO_ROOT = Path(__file__).resolve().parents[1]  # tools/의 한 단계 위 = 레포 루트

MODEL_PATH = REPO_ROOT / "ml" / "artifacts" / "xgb_model.json"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"

# 과신 완화(표시용 분포 평탄화)
TEMPERATURE_T = 2.5
# Top-K 후보 재분배(표시용). True면 Top-K만 재정규화(+floor)해서 "후보군"을 더 고르게 보여줌
RENORMALIZE_TOPK = True
TOPK_FLOOR = 0.02

DEFAULT_TOPK = 5

# =========================
# 2) 내부 캐시(최초 1회 로드)
# =========================
_BOOSTER: Optional[xgb.Booster] = None
_FEATURE_NAMES: Optional[List[str]] = None
_CLASSES: Optional[List[str]] = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_loaded() -> None:
    global _BOOSTER, _FEATURE_NAMES, _CLASSES
    if _BOOSTER is not None and _FEATURE_NAMES is not None and _CLASSES is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {MODEL_PATH.resolve()}")
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"artifacts 디렉토리를 찾을 수 없음: {ARTIFACTS_DIR.resolve()}")

    feat = _load_json(ARTIFACTS_DIR / "feature_names.json")
    lab = _load_json(ARTIFACTS_DIR / "label_mapping.json")

    _FEATURE_NAMES = list(feat["feature_names"])
    _CLASSES = list(lab["classes"])

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    _BOOSTER = booster


def _apply_temperature_on_proba(proba: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    """p' ∝ p^(1/T). T>1 => flatter."""
    P = np.clip(np.asarray(proba), eps, 1.0)
    P_t = P ** (1.0 / T)
    return P_t / P_t.sum(axis=1, keepdims=True)


def _topk_redistribute_row(p_row: np.ndarray, k: int, floor: float, eps: float = 1e-12):
    """Top-K만 뽑아 재정규화 + floor 적용(표시용 확률)."""
    p = np.clip(np.asarray(p_row), eps, 1.0)
    k = min(k, p.shape[0])
    idx = np.argpartition(p, -k)[-k:]
    idx = idx[np.argsort(p[idx])[::-1]]
    pk = p[idx]
    pk = pk / pk.sum()
    if floor and floor > 0:
        pk = np.maximum(pk, floor)
        pk = pk / pk.sum()
    return idx, pk


def _build_vector_from_symptoms(symptoms: Sequence[str], feature_names: Sequence[str]) -> np.ndarray:
    """증상 피처명 리스트 -> (1, F) 0/1 벡터"""
    idx_map = {f: i for i, f in enumerate(feature_names)}
    x = np.zeros((1, len(feature_names)), dtype=np.float32)
    for s in symptoms:
        s = str(s).strip()
        if s and s in idx_map:
            x[0, idx_map[s]] = 1.0
    return x


def predict_topk_diseases(symptoms: List[str], topk: int = DEFAULT_TOPK) -> List[str]:
    """
    입력: 증상 피처명 리스트(list[str])
    출력: Top-K 질병명 리스트(list[str])

    예)
      predict_topk_diseases(["headache","nausea"], topk=5)
        -> ["labyrinthitis", "common cold", ...]
    """
    _ensure_loaded()
    assert _BOOSTER is not None and _FEATURE_NAMES is not None and _CLASSES is not None

    x = _build_vector_from_symptoms(symptoms, _FEATURE_NAMES)
    dm = xgb.DMatrix(x, feature_names=_FEATURE_NAMES)
    proba = _BOOSTER.predict(dm)  # (1, C)

    # 1) temperature scaling (전체 분포 완만화)
    proba = _apply_temperature_on_proba(proba, T=TEMPERATURE_T)

    # 2) topk 선택
    if RENORMALIZE_TOPK:
        idx, _ = _topk_redistribute_row(proba[0], k=topk, floor=TOPK_FLOOR)
    else:
        idx = np.argsort(proba[0])[-topk:][::-1]

    return [_CLASSES[int(i)] for i in idx]


# alias (짧게 쓰고 싶으면)
predict = predict_topk_diseases


@dataclass
class MLPredictTool:
    """
    Orchestrator 호환 래퍼.
    - 입력: symptoms(list[str])
    - 출력: [{"label": str, "score": float}, ...]
    """
    topk: int = DEFAULT_TOPK

    def predict(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        labels = predict_topk_diseases(symptoms, topk=self.topk)
        # 현재 함수는 확률을 반환하지 않으므로 score는 임시 0.0
        return [{"label": lb, "score": 0.0} for lb in labels]
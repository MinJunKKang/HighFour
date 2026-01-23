"""
평가 지표 모음: Hit@K, MRR, Macro F1, Entropy, Confidence, ECE(선택)
- 멀티클래스 확률 출력(proba: [N, C]) 기준
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def hit_at_k(y_true: np.ndarray, proba: np.ndarray, k: int = 5) -> float:
    """Hit@K (=Top-K Accuracy)."""
    y_true = np.asarray(y_true).reshape(-1)
    P = np.asarray(proba)
    if k >= P.shape[1]:
        return 1.0
    topk = np.argpartition(P, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == y_true[:, None], axis=1)))


def mrr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """MRR(Mean Reciprocal Rank)."""
    y_true = np.asarray(y_true).reshape(-1)
    order = np.argsort(-np.asarray(proba), axis=1)
    ranks = np.where(order == y_true[:, None])[1] + 1
    return float(np.mean(1.0 / ranks))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def entropy_of_prediction(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """샘플별 엔트로피 H(p) = -sum p log p (nats)."""
    P = np.clip(np.asarray(proba), eps, 1.0)
    return -np.sum(P * np.log(P), axis=1)


def mean_top1_confidence(proba: np.ndarray) -> float:
    """Top-1 확신도(최대 확률)의 평균."""
    return float(np.max(np.asarray(proba), axis=1).mean())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE(Expected Calibration Error)."""
    conf = np.asarray(conf).reshape(-1)
    correct = np.asarray(correct).astype(int).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            m = (conf >= lo) & (conf < hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.mean()) * abs(acc_bin - conf_bin)
    return float(ece)

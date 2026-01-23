"""
데이터 로드/전처리/분리 모듈

- CSV에서 diseases(라벨) + 나머지 증상 피처(0/1)를 읽는다.
- 희귀 클래스는 __RARE__로 묶을 수 있다.
- stratify split 문제(클래스 1개짜리 등)를 피하기 위해 커스텀 stratified split 사용.

저장 산출물(artifacts):
- label_mapping.json : classes(인덱스->라벨명), rare_label, min_count
- feature_names.json : 피처명 리스트
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    classes: List[str]  # index -> disease name
    rare_label: str
    min_count_per_class: int


def _label_encode(y_str: pd.Series) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    classes = sorted(pd.unique(y_str.astype(str)))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls2id[v] for v in y_str.astype(str)], dtype=np.int32)
    return y, classes, cls2id


def _stratified_split_indices(y: np.ndarray, test_size: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 클래스별로 비율에 맞게 test를 뽑는 커스텀 split.
    - 각 클래스에 최소 1개는 test로 간다(가능한 경우).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            n_test = 1
        else:
            n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def make_train_val_test(
    y: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    train/val/test 인덱스 반환.
    val_size는 전체 기준(예: 0.1)이므로 train 내부에서 비율로 환산한다.
    """
    train_idx, test_idx = _stratified_split_indices(y, test_size=test_size, seed=seed)
    y_train = y[train_idx]
    val_ratio_in_train = val_size / (1.0 - test_size)
    tr2_rel, val_rel = _stratified_split_indices(y_train, test_size=val_ratio_in_train, seed=seed + 1)
    val_idx = train_idx[val_rel]
    final_train_idx = train_idx[tr2_rel]
    return final_train_idx, val_idx, test_idx


def load_and_split(
    csv_path: str | Path,
    artifacts_dir: str | Path,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_seed: int = 42,
    min_count_per_class: int = 10,
    rare_label: str = "__RARE__",
) -> SplitData:
    csv_path = Path(csv_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "diseases" not in df.columns:
        raise ValueError("CSV에 'diseases' 컬럼이 필요합니다.")

    y_raw = df["diseases"].astype(str)
    X_df = df.drop(columns=["diseases"])

    # 0/1 데이터면 uint8로 축소
    vals = pd.unique(X_df.values.ravel())
    ok01 = True
    for v in vals[:1000]:
        if pd.isna(v):
            continue
        try:
            iv = int(v)
        except Exception:
            ok01 = False
            break
        if iv not in (0, 1):
            ok01 = False
            break
    X_df = X_df.astype(np.uint8 if ok01 else np.float32)

    # 희귀 클래스 묶기
    vc = y_raw.value_counts()
    rare = vc[vc < min_count_per_class].index
    y_adj = y_raw.where(~y_raw.isin(rare), other=rare_label)

    y, classes, _ = _label_encode(y_adj)
    feature_names = list(X_df.columns)

    tr_idx, va_idx, te_idx = make_train_val_test(y, test_size=test_size, val_size=val_size, seed=random_seed)
    X = X_df.values

    split = SplitData(
        X_train=X[tr_idx], y_train=y[tr_idx],
        X_val=X[va_idx], y_val=y[va_idx],
        X_test=X[te_idx], y_test=y[te_idx],
        feature_names=feature_names,
        classes=classes,
        rare_label=rare_label,
        min_count_per_class=int(min_count_per_class),
    )

    (artifacts_dir / "label_mapping.json").write_text(
        json.dumps(
            {"classes": classes, "rare_label": rare_label, "min_count_per_class": int(min_count_per_class)},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    (artifacts_dir / "feature_names.json").write_text(
        json.dumps({"feature_names": feature_names}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return split

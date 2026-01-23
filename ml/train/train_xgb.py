"""
XGBoost 학습 스크립트 (GPU 지원)

실행 예시:
python -m ml.train.train_xgb --csv Final_Augmented_dataset_Diseases_and_Symptoms.csv --outdir ml/artifacts --gpu

산출물(ml/artifacts):
- xgb_model.json
- label_mapping.json
- feature_names.json
- train_config.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from split import load_and_split
from eval_metrics import hit_at_k, mrr, macro_f1, entropy_of_prediction, mean_top1_confidence


def symptom_dropout(X: np.ndarray, drop_p: float = 0.10, seed: int = 42) -> np.ndarray:
    """학습 데이터에만 적용: 1인 피처를 확률 drop_p로 0으로 만든다(과신 완화)."""
    rng = np.random.RandomState(seed)
    X2 = X.copy()
    mask = (X2 > 0) & (rng.rand(*X2.shape) < drop_p)
    X2[mask] = 0
    return X2


def train(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    split = load_and_split(
        csv_path=args.csv,
        artifacts_dir=outdir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
        min_count_per_class=args.min_count,
        rare_label=args.rare_label,
    )

    num_classes = len(split.classes)

    X_train = split.X_train
    if args.symptom_drop_p > 0:
        X_train = symptom_dropout(X_train, drop_p=args.symptom_drop_p, seed=args.seed)

    dtrain = xgb.DMatrix(X_train, label=split.y_train, feature_names=split.feature_names)
    dval   = xgb.DMatrix(split.X_val, label=split.y_val, feature_names=split.feature_names)
    dtest  = xgb.DMatrix(split.X_test, label=split.y_test, feature_names=split.feature_names)

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": num_classes,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.l2,
        "min_child_weight": args.min_child_weight,
        "max_bin": args.max_bin,
        "seed": args.seed,
    }

    if args.gpu:
        params.update({"tree_method": "hist", "device": "cuda"})
    else:
        params.update({"tree_method": "hist"})

    watchlist = [(dtrain, "train"), (dval, "val")]

    t0 = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=watchlist,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    train_time = time.time() - t0

    # test 평가
    proba_test = booster.predict(dtest)
    y_pred = np.argmax(proba_test, axis=1)

    metrics = {
        "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
        "best_score": float(booster.best_score) if booster.best_score is not None else None,
        "hit@1": hit_at_k(split.y_test, proba_test, k=1),
        "hit@3": hit_at_k(split.y_test, proba_test, k=3),
        "hit@5": hit_at_k(split.y_test, proba_test, k=5),
        "mrr": mrr(split.y_test, proba_test),
        "macro_f1": macro_f1(split.y_test, y_pred),
        "entropy_mean": float(entropy_of_prediction(proba_test).mean()),
        "top1_conf_mean": mean_top1_confidence(proba_test),
        "train_time_sec": float(train_time),
        "num_classes": int(num_classes),
        "train_n": int(len(split.y_train)),
        "val_n": int(len(split.y_val)),
        "test_n": int(len(split.y_test)),
    }

    print("\n===== TEST METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    model_path = outdir / "xgb_model.json"
    booster.save_model(str(model_path))

    cfg = {
        "csv": str(Path(args.csv).resolve()),
        "outdir": str(outdir.resolve()),
        "params": params,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "symptom_drop_p": args.symptom_drop_p,
        "metrics": metrics,
    }
    (outdir / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {model_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="학습 CSV 경로")
    p.add_argument("--outdir", default="ml/artifacts", help="산출물 저장 경로")
    p.add_argument("--gpu", action="store_true", help="GPU 사용(device=cuda)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.10)

    p.add_argument("--min_count", type=int, default=10, help="희귀 클래스 최소 빈도(미만은 RARE로 묶음)")
    p.add_argument("--rare_label", type=str, default="__RARE__")

    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--eta", type=float, default=0.10)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--min_child_weight", type=float, default=5.0)
    p.add_argument("--max_bin", type=int, default=256)

    p.add_argument("--num_boost_round", type=int, default=2000)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--verbose_eval", type=int, default=50)

    p.add_argument("--symptom_drop_p", type=float, default=0.10, help="학습 입력 증상 드랍 비율(0이면 비활성)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

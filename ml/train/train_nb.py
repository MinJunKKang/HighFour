# 실행 명령어
# $ python ml/train/train_nb.py --csv ml/train/Final_Augmented_dataset_Diseases_and_Symptoms.csv

"""
Bernoulli Naive Bayes 학습
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB


from split import load_and_split
from eval_metrics import hit_at_k, mean_top1_confidence


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

    
    X_train = np.nan_to_num(split.X_train, nan=0.0)
    X_test = np.nan_to_num(split.X_test, nan=0.0)

    model = BernoulliNB(alpha=args.alpha)


    t0 = time.time()
    model.fit(X_train, split.y_train)
    train_time = time.time() - t0

    
    print("지표 계산 중")
    proba_test = model.predict_proba(X_test)

    metrics = {
        "hit@1": hit_at_k(split.y_test, proba_test, k=1),
        "hit@3": hit_at_k(split.y_test, proba_test, k=3),
        "hit@5": hit_at_k(split.y_test, proba_test, k=5),
        "top1_conf_mean": mean_top1_confidence(proba_test),
        "train_time_sec": float(train_time),
        "num_classes": int(len(split.classes)),
        "test_n": int(len(split.y_test)),
    }

    print("\n===== NB TEST METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    
    model_path = outdir / "nb_model.json"
    cfg = {
        "model_type": "BernoulliNB",
        "params": model.get_params(),
        "metrics": metrics,
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    
    print(f"\n[saved] {model_path}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="../artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--rare_label", default="__RARE__")
    p.add_argument("--alpha", type=float, default=1.0)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
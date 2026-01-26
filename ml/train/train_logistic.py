# 실행 명령어
# python -m ml.train.train_logistic --csv ml/train/Final_Augmented_dataset_Diseases_and_Symptoms.csv
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression

from .split import load_and_split
from .eval_metrics import hit_at_k, mrr, macro_f1, entropy_of_prediction, mean_top1_confidence

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

    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.seed,
        solver='lbfgs'
    )

    t0 = time.time()
    model.fit(split.X_train, split.y_train)
    train_time = time.time() - t0

    proba_test = model.predict_proba(split.X_test)
    y_pred = model.predict(split.X_test)

    metrics = {
        "hit@1": hit_at_k(split.y_test, proba_test, k=1),
        "hit@3": hit_at_k(split.y_test, proba_test, k=3),
        "hit@5": hit_at_k(split.y_test, proba_test, k=5),
        "mrr": mrr(split.y_test, proba_test),
        "macro_f1": macro_f1(split.y_test, y_pred),
        "entropy_mean": float(entropy_of_prediction(proba_test).mean()),
        "top1_conf_mean": mean_top1_confidence(proba_test),
        "train_time_sec": float(train_time)
    }

    print("\n===== LOGISTIC REGRESSION 결과 =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    
    model_json_path = outdir / "logistic_model.json"
    model_data = {
        "model_type": "LogisticRegression",
        "classes": list(split.classes),
        "feature_names": split.feature_names,
        "intercept": model.intercept_.tolist(),
        "coef": model.coef_.tolist()
    }
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

    # 설명 에이전트(Explain Agent)용 가중치 데이터 생성 (기존 유지)
    coef_dict = {}
    for i, class_name in enumerate(split.classes):
        top_indices = np.argsort(np.abs(model.coef_[i]))[-10:][::-1]
        coef_dict[class_name] = {
            split.feature_names[idx]: float(model.coef_[i][idx])
            for idx in top_indices
        }

    with open(outdir / "feature_importance_logistic.json", "w", encoding="utf-8") as f:
        json.dump(coef_dict, f, ensure_ascii=False, indent=2)

    cfg = {
        "model_type": "LogisticRegression_OvR",
        "params": model.get_params(),
        "metrics": metrics
    }
    (outdir / "train_config_logistic.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\n[Saved] {model_json_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="학습 CSV 경로")
    p.add_argument("--outdir", default="ml/artifacts", help="산출물 저장 경로")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--rare_label", type=str, default="__RARE__")
    p.add_argument("--C", type=float, default=1.0, help="규제 강도")
    p.add_argument("--max_iter", type=int, default=1000)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
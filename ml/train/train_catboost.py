# catboost 설치
# pip install catboost

# 실행 명령어
# $ python -m ml.train.train_catboost --csv ml/train/Final_Augmented_dataset_Diseases_and_Symptoms.csv

"""
CatBoost 학습 스크립트
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .split import load_and_split
from .eval_metrics import hit_at_k, mrr, macro_f1, entropy_of_prediction, mean_top1_confidence

def _detect_catboost_gpu() -> bool:
    """CUDA 기반 CatBoost GPU 사용 가능 여부를 최대한 안전하게 감지."""
    try:
        from catboost.utils import get_gpu_device_count  # type: ignore
        return int(get_gpu_device_count()) > 0
    except Exception:
        # (1) CPU 전용 빌드이거나 (2) 드라이버/CUDA 미설치 등으로 조회 실패하는 경우
        return False



def train(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

   
    raw_df = pd.read_csv(args.csv)
    if raw_df.isna().sum().sum() > 0:
        print(f">>> NaN 데이터 발견 ({raw_df.isna().sum().sum()}개). 자동 정제 중...")
        raw_df = raw_df.fillna(0)
        clean_csv = Path(args.csv).parent / "temp_clean_dataset.csv"
        raw_df.to_csv(clean_csv, index=False)
        target_csv = str(clean_csv)
    else:
        target_csv = args.csv

   
    split = load_and_split(
        csv_path=target_csv,
        artifacts_dir=outdir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
        min_count_per_class=args.min_count,
        rare_label=args.rare_label,
    )


    params = {
        "loss_function": "MultiClass",
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.lr,
        "random_seed": args.seed,
        "verbose": 50,
        "allow_writing_files": False 
    }
    
    # GPU 사용 정책
    # - 기본: GPU가 있으면 자동으로 GPU 사용
    # - --gpu: GPU 강제 (실패 시 에러로 중단)
    # - --force_cpu: CPU 강제
    use_gpu = False
    mode = "CPU"
    if args.force_cpu:
        use_gpu = False
        mode = "CPU(forced)"
    elif args.gpu:
        use_gpu = True
        mode = f"GPU(forced, devices={args.devices})"
    else:
        use_gpu = _detect_catboost_gpu()
        mode = f"GPU(auto, devices={args.devices})" if use_gpu else "CPU(auto)"

    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = args.devices

    model = CatBoostClassifier(**params)

    print(f">>> CatBoost 실행 모드: {mode}")

    t0 = time.time()
    try:
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=(split.X_val, split.y_val),
            early_stopping_rounds=50
        )
    except Exception as e:
        # 자동 모드에서만: GPU 초기화/학습 실패 시 CPU로 폴백
        if use_gpu and (not args.gpu) and (not args.force_cpu):
            print(f">>> GPU 학습 실패 감지 → CPU로 재시도합니다. (원인: {type(e).__name__}: {e})")
            params.pop("task_type", None)
            params.pop("devices", None)
            model = CatBoostClassifier(**params)
            print(">>> CatBoost 실행 모드: CPU(fallback)")
            model.fit(
                split.X_train,
                split.y_train,
                eval_set=(split.X_val, split.y_val),
                early_stopping_rounds=50
            )
        else:
            raise
    train_time = time.time() - t0


    print("지표 계산 중")
    proba_test = model.predict_proba(split.X_test)
    y_pred = np.argmax(proba_test, axis=1)

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

    print("\n===== CATBOOST 결과 =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    # =========================
    # 모델 저장 (권장: CBM + 순수 CatBoost JSON + 메타 분리)
    # - catboost_model.cbm  : 가장 안정/호환성 높음
    # - catboost_model.json : CatBoost가 load_model(format="json")로 직접 읽는 '순수 모델 JSON'
    # - catboost_meta.json  : 클래스/피처/파라미터/지표 등 메타데이터 (모델 JSON을 감싸지 않기!)
    # =========================
    model_cbm_path = outdir / "catboost_model.cbm"
    model_raw_json_path = outdir / "catboost_model.json"
    meta_json_path = outdir / "catboost_meta.json"

    # 1) 바이너리(.cbm) 저장
    model.save_model(str(model_cbm_path))

    # 2) (옵션) 순수 모델 JSON 저장
    model.save_model(str(model_raw_json_path), format="json")

    # 3) 메타데이터 저장 (별도 파일)
    meta = {
        "model_type": "CatBoost",
        "classes": list(getattr(split, "classes", [])),
        "feature_names": list(getattr(split, "feature_names", [])),
        "rare_label": getattr(split, "rare_label", None),
        "min_count_per_class": getattr(split, "min_count_per_class", None),
        "params": params if "params" in locals() else None,
        "metrics": metrics if "metrics" in locals() else None,
    }
    meta_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    importance = model.get_feature_importance()
    importance_dict = {}
   
    top_indices = np.argsort(importance)[-10:][::-1]
    global_top_features = {
        split.feature_names[idx]: float(importance[idx])
        for idx in top_indices
    }
    for class_name in split.classes:
        importance_dict[class_name] = global_top_features

    with open(outdir / "feature_importance_catboost.json", "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, ensure_ascii=False, indent=2)

    cfg = {
        "model_type": "CatBoost",
        "params": params,
        "metrics": metrics
    }
    (outdir / "train_config_catboost.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    
    print(f"\n[Saved] 3개의 산출물이 {outdir}에 생성되었습니다.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="ml/artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--rare_label", default="__RARE__")
    
    # CatBoost 전용 하이퍼파라미터
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.1)

    # GPU 관련 옵션
    # - 기본 동작: GPU가 있으면 자동으로 GPU 사용
    # - --gpu: GPU 강제 (실패 시 에러)
    # - --force_cpu: CPU 강제
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--devices", default="0", help='CatBoost GPU device id(s), 예: "0" 또는 "0:1"')

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
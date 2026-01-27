# 설치
# pip install scikit-learn joblib
#
# (GPU 사용을 원하면) RAPIDS cuML 필요:
#   - pip/conda 환경에 따라 설치 방식이 다를 수 있음
#   - 설치가 안 되어 있으면 자동으로 CPU(sklearn)로 폴백됨
#
# 실행
# python -m ml.train.train_rf --csv ml/train/Final_Augmented_dataset_Diseases_and_Symptoms.csv

"""
Random Forest 학습 스크립트
- 기본: GPU가 가능하면 GPU(cuML) 사용 → 실패/미설치면 CPU(sklearn)로 폴백
- --gpu        : GPU 강제(실패 시 에러)
- --force_cpu  : CPU 강제
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .split import load_and_split
from .eval_metrics import hit_at_k, mrr, macro_f1, entropy_of_prediction, mean_top1_confidence


def _gpu_device_count() -> int:
    """CUDA GPU 존재 여부(최소한의 체크)."""
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _cuml_rf_available() -> bool:
    """cuML RandomForestClassifier 사용 가능 여부."""
    try:
        from cuml.ensemble import RandomForestClassifier as _  # type: ignore
        return True
    except Exception:
        return False


def _to_numpy(x):
    """cupy/numba/기타 배열을 numpy로 변환."""
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def train(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Split 로딩 정책
    # - --split 을 주면: 해당 split JSON(학습/평가 공통)을 그대로 사용
    # - 없으면: 기존 방식(load_and_split)로 새로 생성
    # -----------------------------
    if args.split is not None:
        sp = json.loads(Path(args.split).read_text(encoding="utf-8"))

        # split 파일이 가진 설정을 우선 사용(학습/평가 일치 보장)
        feature_names = sp["feature_names"]
        classes = sp["classes"]
        rare_label = sp.get("rare_label", "__RARE__")
        min_count = int(sp.get("min_count", sp.get("min_count_per_class", 10)))

        # 인덱스 키는 여러 형태를 지원
        def _get_idx(key_candidates):
            for k in key_candidates:
                if k in sp:
                    return np.asarray(sp[k], dtype=np.int64)
            return None

        idx_train = _get_idx(["idx_train", "train_idx", "train_indices"])
        idx_val   = _get_idx(["idx_val", "val_idx", "val_indices"])
        idx_test  = _get_idx(["idx_test", "test_idx", "test_indices"])

        if idx_train is None or idx_val is None or idx_test is None:
            raise ValueError(
                f"split JSON에 idx_train/idx_val/idx_test가 필요합니다. "
                f"(현재: {sorted(list(sp.keys()))})"
            )

        df = pd.read_csv(args.csv)

        # split과 동일한 rare 처리(평가 코드와 동일한 방식)
        y_raw = df[args.target].astype(str)
        counts = y_raw.value_counts()
        y_mapped = y_raw.where(y_raw.map(counts) >= min_count, other=rare_label)

        le = LabelEncoder()
        le.fit([str(c) for c in classes])
        y = le.transform(y_mapped.astype(str)).astype(np.int32)

        X_df = df[feature_names].copy()
        # 0/1이면 uint8로 (메모리/속도)
        ok01 = True
        for c in X_df.columns:
            v = X_df[c]
            if v.isna().any():
                ok01 = False
                break
            if not set(pd.unique(v)).issubset({0, 1}):
                ok01 = False
                break
        X_df = X_df.astype(np.uint8 if ok01 else np.float32)

        X = X_df.values

        split = type("Tmp", (), {})()
        split.X_train, split.y_train = X[idx_train], y[idx_train]
        split.X_val,   split.y_val   = X[idx_val],   y[idx_val]
        split.X_test,  split.y_test  = X[idx_test],  y[idx_test]
        split.feature_names = list(feature_names)
        split.classes = list(classes)
        split.rare_label = str(rare_label)
        split.min_count_per_class = int(min_count)

        # artifacts를 split 파일 기준으로 저장(일관성)
        (outdir / "label_mapping.json").write_text(
            json.dumps(
                {"classes": split.classes, "rare_label": split.rare_label, "min_count_per_class": split.min_count_per_class},
                ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        (outdir / "feature_names.json").write_text(
            json.dumps({"feature_names": split.feature_names}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        split = load_and_split(
            csv_path=args.csv,
            artifacts_dir=outdir,
            test_size=args.test_size,
            val_size=args.val_size,
            random_seed=args.seed,
            min_count_per_class=args.min_count,
            rare_label=args.rare_label,
        )

    # ---------- GPU/CPU 선택 정책 ----------
    want_gpu = False
    if args.force_cpu:
        want_gpu = False
        mode = "CPU(forced)"
    elif args.gpu:
        want_gpu = True
        mode = "GPU(forced)"
    else:
        # 자동: GPU가 "실제로" 보이고 + cuML이 import 되면 GPU 시도
        want_gpu = (_gpu_device_count() > 0) and _cuml_rf_available()
        mode = "GPU(auto)" if want_gpu else "CPU(auto)"

    # 공통 파라미터(가능한 범위 내에서 통일)
    common_kwargs = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        max_features=args.max_features,
        bootstrap=not args.no_bootstrap,
        random_state=args.seed,
    )

    # sklearn에서만 의미 있는 옵션
    sklearn_kwargs = dict(
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
    )

    model = None
    backend = None

    # ---------- 학습 ----------
    print(f">>> RandomForest 실행 모드: {mode}")

    t0 = time.time()
    if want_gpu:
        try:
            # cuML GPU RF
            import cupy as cp  # type: ignore
            from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore

            X_train = cp.asarray(np.asarray(split.X_train))
            y_train = cp.asarray(np.asarray(split.y_train))
            X_val = cp.asarray(np.asarray(split.X_val))
            y_val = cp.asarray(np.asarray(split.y_val))
            X_test = cp.asarray(np.asarray(split.X_test))

            # cuML은 sklearn과 파라미터 지원 범위가 다를 수 있어 최소한으로 전달
            model = cuRF(**common_kwargs)
            backend = "cuml"

            model.fit(X_train, y_train)

            train_time = time.time() - t0

            print("지표 계산 중")
            # cuML: predict/predict_proba 결과가 cupy일 수 있음
            proba_test = _to_numpy(model.predict_proba(X_test))
            y_pred = _to_numpy(model.predict(X_test))

        except Exception as e:
            # 자동 모드에서만 폴백
            if (not args.gpu) and (not args.force_cpu):
                print(f">>> GPU 학습 실패 감지 → CPU(sklearn)로 재시도합니다. (원인: {type(e).__name__}: {e})")
                want_gpu = False
                mode = "CPU(fallback)"
            else:
                raise

    if not want_gpu:
        from sklearn.ensemble import RandomForestClassifier as skRF

        model = skRF(**common_kwargs, **sklearn_kwargs)
        backend = "sklearn"

        model.fit(split.X_train, split.y_train)
        train_time = time.time() - t0

        print("지표 계산 중")
        proba_test = model.predict_proba(split.X_test)
        y_pred = model.predict(split.X_test)

    # ---------- 메트릭 ----------
    metrics = {
        "backend": backend,
        "hit@1": hit_at_k(split.y_test, proba_test, k=1),
        "hit@3": hit_at_k(split.y_test, proba_test, k=3),
        "hit@5": hit_at_k(split.y_test, proba_test, k=5),
        "mrr": mrr(split.y_test, proba_test),
        "macro_f1": macro_f1(split.y_test, y_pred),
        "entropy_mean": float(entropy_of_prediction(proba_test).mean()),
        "top1_conf_mean": mean_top1_confidence(proba_test),
        "train_time_sec": float(train_time),
    }

    print("\n===== RANDOM FOREST 결과 =====")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ---------- 저장 ----------
    model_json_path = outdir / "rf_model.json"

    # 모델 메타(JSON) — eval에서 classes/feature_names 매칭용
    model_full_data = {
        "model_type": "RandomForest",
        "backend": backend,
        "classes": list(split.classes),
        "feature_names": split.feature_names,
        "params": getattr(model, "get_params", lambda: {})(),
    }
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_full_data, f, ensure_ascii=False, indent=2)

    # 모델 파일 저장(pkl)
    # - sklearn: joblib.dump가 안정적
    # - cuml: 환경에 따라 직렬화가 실패할 수 있어 try/except
    if backend == "sklearn":
        joblib.dump(model, outdir / "rf_model.pkl")
    else:
        try:
            joblib.dump(model, outdir / "rf_model_cuml.pkl")
        except Exception as e:
            print(f">>> cuML 모델 직렬화(joblib)가 실패했습니다. (원인: {type(e).__name__}: {e})")
            print(">>> 이 경우 rf_model.json + 지표 파일은 생성되며, 모델 재사용은 별도 저장 방식이 필요할 수 있습니다.")

    # Feature importance(가능한 경우)
    try:
        importances = _to_numpy(getattr(model, "feature_importances_"))
        top_indices = np.argsort(importances)[-10:][::-1]
        global_top_features = {
            split.feature_names[idx]: float(importances[idx])
            for idx in top_indices
        }
        importance_dict = {class_name: global_top_features for class_name in split.classes}

        with open(outdir / "feature_importance_rf.json", "w", encoding="utf-8") as f:
            json.dump(importance_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f">>> feature_importances 저장을 건너뜁니다. (원인: {type(e).__name__}: {e})")

    cfg = {
        "model_type": "RandomForestClassifier",
        "backend": backend,
        "params": model_full_data["params"],
        "metrics": metrics,
    }
    (outdir / "train_config_rf.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"\n[Saved] 산출물이 {outdir}에 생성되었습니다.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", default="diseases")
    p.add_argument("--split", default=None, help="eval_splits에서 만든 split JSON 경로(학습/평가 완전 일치용)")
    p.add_argument("--outdir", default="ml/artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--rare_label", default="__RARE__")

    # RF 전용 파라미터
    p.add_argument("--n_estimators", type=int, default=400)
    p.add_argument("--max_depth", type=int, default=26)
    p.add_argument("--min_samples_split", type=int, default=6)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--max_features", default="sqrt")
    p.add_argument("--no_bootstrap", action="store_true")
    p.add_argument("--class_weight", default="balanced_subsample")
    p.add_argument("--n_jobs", type=int, default=-1)

    # GPU 관련 옵션
    p.add_argument("--gpu", action="store_true", help="GPU 강제(cuML 필요). 실패 시 에러")
    p.add_argument("--force_cpu", action="store_true", help="CPU 강제(sklearn)")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

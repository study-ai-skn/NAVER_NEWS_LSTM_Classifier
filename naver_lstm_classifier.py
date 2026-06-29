"""Optuna 튜닝 → 최적 파라미터로 학습까지 일괄 처리하는 진입점.

실행 예:
  python naver_lstm_classifier.py                          # LSTM/KoBERT/KoELECTRA 전부
  python naver_lstm_classifier.py --models LSTM            # LSTM 만
  python naver_lstm_classifier.py --models KoBERT KoELECTRA  # 트랜스포머만
  python naver_lstm_classifier.py --models LSTM --n_trials 20

실행 순서:
  1. models/v{N}/ 버전 폴더 자동 생성
  2. 지정 모델 유형에 대해 Optuna 탐색
  3. best_configs.json 저장
  4. 최적 파라미터로 각 모델 학습
  5. 산출물(모델, 시각화) 저장
"""

import argparse
import re
from pathlib import Path

from app.predict import predict_text
from app.train import train_model
from app.tune import run_tuning

_MODELS_DIR = Path(__file__).parent / "models"


def _next_run_dir() -> Path:
    existing = sorted(
        int(d.name[1:])
        for d in _MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^v\d+$", d.name)
    ) if _MODELS_DIR.exists() else []
    version = (existing[-1] + 1) if existing else 1
    run_dir = _MODELS_DIR / f"v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna 튜닝 + 학습 일괄 실행")
    parser.add_argument(
        "--models", nargs="+",
        default=["LSTM", "KoBERT", "KoELECTRA"],
        choices=["LSTM", "KoBERT", "KoELECTRA"],
        help="학습할 모델 유형 (기본: 전부)",
    )
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trial 수")
    args = parser.parse_args()

    run_dir = _next_run_dir()
    print(f"\n{'='*60}")
    print(f"  버전: {run_dir.name}  ({run_dir})")
    print(f"  모델: {args.models}")
    print(f"{'='*60}\n")

    # ── 1. Optuna 탐색 ────────────────────────────────────────────────────────
    best_configs = run_tuning(
        n_trials=args.n_trials,
        save_dir=str(run_dir),
        model_types=args.models,
    )

    # ── 2. 최적 파라미터로 학습 ───────────────────────────────────────────────
    results = {}
    sample_news = "삼성전자 인공지능 반도체 기술 개발 성공 발표"

    for model_type in args.models:
        if model_type not in best_configs:
            print(f"[{model_type}] Optuna 결과 없음 — 기본 파라미터로 건너뜀")
            continue

        cfg = best_configs[model_type]
        cfg.model_path = str(run_dir / f"naver_{model_type.lower()}_model.pt")

        print(f"\n{'='*60}")
        print(f"  [{model_type}] 최적 파라미터로 최종 학습")
        print(f"{'='*60}")
        model, metadata = train_model(cfg)
        results[model_type] = metadata["accuracy"]

        pred = predict_text(sample_news, model, metadata, cfg)
        print(f"  예측: '{sample_news}' → {pred}")

    # ── 3. 결과 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [{run_dir.name}] 최종 결과")
    print(f"{'='*60}")
    for mt, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(acc * 20)
        print(f"  {mt:12s}: {acc:.4f}  {bar}")
    print(f"\n  산출물 위치: {run_dir}")
    print(f"{'='*60}\n")

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
import json
import re
from pathlib import Path

from app.config import Config
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
    parser.add_argument("--max-items", type=int, default=300, help="카테고리당 크롤링 건수 (기본 300)")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Optuna 탐색 건너뛰고 최신 best_configs.json 으로 바로 학습")
    parser.add_argument("--epochs", type=int, default=None, help="학습 epoch 수 (기본: config 값 사용)")
    parser.add_argument("--patience", type=int, default=None, help="Early Stop patience 수")
    parser.add_argument("--lr", type=float, default=None, help="학습률 직접 지정 (config 값 덮어씀)")
    args = parser.parse_args()

    run_dir = _next_run_dir()
    print(f"\n{'='*60}")
    print(f"  버전: {run_dir.name}  ({run_dir})")
    print(f"  모델: {args.models}")
    print(f"{'='*60}\n")

    # ── 1. Optuna 탐색 (또는 기존 결과 로드) ─────────────────────────────────
    if args.skip_tuning:
        # 최신 버전 폴더에서 best_configs.json 로드
        existing = sorted(
            int(d.name[1:])
            for d in _MODELS_DIR.iterdir()
            if d.is_dir() and re.match(r"^v\d+$", d.name) and d != run_dir
        )
        best_configs = {}
        for v in reversed(existing):
            cfg_path = _MODELS_DIR / f"v{v}" / "best_configs.json"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as f:
                    data = json.load(f)
                for mt, params in data.items():
                    if mt in args.models:
                        params.pop("model_path", None)
                        best_configs[mt] = Config(**params)
                print(f"  기존 best_configs 로드: {cfg_path}")
                break
        if not best_configs:
            print("  [경고] best_configs.json 없음 — 기본 파라미터 사용")
            best_configs = {mt: Config(model_type=mt) for mt in args.models}
    else:
        best_configs = run_tuning(
            n_trials=args.n_trials,
            save_dir=str(run_dir),
            model_types=args.models,
            max_items=args.max_items,
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
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.patience is not None:
            cfg.patience = args.patience
        if args.lr is not None:
            cfg.learning_rate = args.lr

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

    # 결과 저장 (run_v6.py 가 읽어서 리포트 자동 생성)
    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"version": run_dir.name, "results": results}, f, ensure_ascii=False, indent=2)

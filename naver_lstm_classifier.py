"""Optuna 하이퍼파라미터 튜닝 전용 진입점.

실행 순서:
  1. models/v{N}/ 버전 폴더 자동 생성
  2. Optuna 로 LSTM / KoBERT / KoELECTRA 세 모델 동시 탐색 (n_trials=30)
  3. 모델별 최적 Config 를 models/v{N}/best_configs.json 에 저장
  4. 시각화 산출물 저장 (optuna_results.png)

튜닝 완료 후 → python main.py 로 세 모델을 순서대로 학습 및 비교
"""

import re
from pathlib import Path

from app.tune import run_tuning

_MODELS_DIR = Path(__file__).parent / "models"


def _next_run_dir() -> Path:
    """기존 models/v* 폴더를 확인해 다음 버전 폴더를 생성하고 반환한다."""
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
    run_dir = _next_run_dir()
    print(f"\n{'='*60}")
    print(f"  Optuna 튜닝 버전: {run_dir.name}  →  {run_dir}")
    print(f"{'='*60}\n")

    # LSTM / KoBERT / KoELECTRA 세 모델을 함께 탐색
    best_configs = run_tuning(n_trials=30, save_dir=str(run_dir))

    print(f"\n{'='*60}")
    print(f"  탐색 완료. 학습을 진행하려면:")
    print(f"  python main.py")
    print(f"{'='*60}\n")

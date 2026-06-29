"""세 모델(LSTM / KoBERT / KoELECTRA) 순차 학습 및 성능 비교 진입점.

실행 전 naver_lstm_classifier.py 로 Optuna 튜닝을 먼저 완료하면
최적 하이퍼파라미터로 각 모델을 학습한다.
튜닝 결과(best_configs.json)가 없으면 기본 하이퍼파라미터를 사용한다.

실행 방법:
  python naver_lstm_classifier.py   # (선택) Optuna 튜닝
  python main.py                    # 세 모델 순차 학습 + 비교
"""

import json
import re
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

from app.config import Config
from app.predict import predict_text
from app.train import train_model

_MODELS_DIR = Path(__file__).parent / "models"

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunsl.ttf",
]
_KOREAN_FONT_PATH: str | None = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
if _KOREAN_FONT_PATH:
    try:
        fm.fontManager.addfont(_KOREAN_FONT_PATH)
        _prop = fm.FontProperties(fname=_KOREAN_FONT_PATH)
        plt.rcParams["font.family"] = _prop.get_name()
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ── 기본 하이퍼파라미터 (Optuna 결과 없을 때 사용) ─────────────────────────────
_DEFAULT_CONFIGS: Dict[str, Config] = {
    "LSTM": Config(
        model_type="LSTM",
    ),
    "KoBERT": Config(
        model_type="KoBERT",
        max_len=128,
        learning_rate=2e-5,
        batch_size=16,
        optimizer_name="AdamW",
        weight_decay=0.01,
        dropout=0.1,
    ),
    "KoELECTRA": Config(
        model_type="KoELECTRA",
        max_len=128,
        learning_rate=2e-5,
        batch_size=16,
        optimizer_name="AdamW",
        weight_decay=0.01,
        dropout=0.1,
    ),
}


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


def _load_best_configs() -> Dict[str, Config]:
    """최신 버전 폴더에서 Optuna 튜닝 결과(best_configs.json)를 불러온다.

    JSON 이 없으면 기본 하이퍼파라미터를 반환한다.
    """
    if not _MODELS_DIR.exists():
        print("Optuna 튜닝 결과 없음 → 기본 하이퍼파라미터 사용")
        return dict(_DEFAULT_CONFIGS)

    existing = sorted(
        int(d.name[1:])
        for d in _MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^v\d+$", d.name)
    )
    for version in reversed(existing):
        cfg_path = _MODELS_DIR / f"v{version}" / "best_configs.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            configs: Dict[str, Config] = {}
            for mt, params in data.items():
                params.pop("model_path", None)
                configs[mt] = Config(**params)
            print(f"Optuna 튜닝 결과 로드: {cfg_path}")
            # 없는 모델 유형은 기본값으로 채운다
            for mt, cfg in _DEFAULT_CONFIGS.items():
                if mt not in configs:
                    print(f"  [{mt}] Optuna 결과 없음 → 기본 하이퍼파라미터 사용")
                    configs[mt] = cfg
            return configs

    print("Optuna 튜닝 결과 없음 → 기본 하이퍼파라미터 사용")
    return dict(_DEFAULT_CONFIGS)


def _plot_comparison(results: Dict[str, Dict], save_dir: str) -> None:
    """세 모델의 테스트 정확도를 비교하는 막대 차트를 저장한다."""
    model_types = list(results.keys())
    accuracies  = [results[mt]["accuracy"]   for mt in model_types]
    val_accs    = [results[mt].get("val_acc", 0.0) for mt in model_types]

    color_map = {"LSTM": "steelblue", "KoBERT": "tomato", "KoELECTRA": "mediumseagreen"}
    colors    = [color_map.get(mt, "gray") for mt in model_types]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(model_types))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.85, label="Test Accuracy")
    ax.scatter(x, val_accs, color="gold", edgecolors="black", zorder=5, s=120, label="Best Val Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(model_types, fontsize=12)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("모델 유형별 최종 성능 비교")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"모델 비교 차트 저장 완료: model_comparison.png")


if __name__ == "__main__":
    run_dir = _next_run_dir()
    print(f"\n{'='*60}")
    print(f"  학습 버전: {run_dir.name}  →  {run_dir}")
    print(f"{'='*60}\n")

    configs = _load_best_configs()

    results: Dict[str, Dict] = {}
    sample_news = "삼성전자 인공지능 반도체 기술 개발 성공 발표"

    for model_type in ["LSTM", "KoBERT", "KoELECTRA"]:
        cfg = configs[model_type]
        cfg.model_path = str(run_dir / f"naver_{model_type.lower()}_model.pt")

        print(f"\n{'='*60}")
        print(f"  [{model_type}] 학습 시작")
        print(f"{'='*60}")
        model, metadata = train_model(cfg)
        accuracy = metadata["accuracy"]
        results[model_type] = {"accuracy": accuracy}

        # 예측 확인
        pred = predict_text(sample_news, model, metadata, cfg)
        print(f"  예측 샘플: '{sample_news}' → {pred}")

    # ── 비교 차트 + 결과 출력 ─────────────────────────────────────────────────
    _plot_comparison(results, str(run_dir))

    print(f"\n{'='*60}")
    print(f"  모델 비교 결과  (버전: {run_dir.name})")
    print(f"{'='*60}")
    for mt, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        bar = "█" * int(r["accuracy"] * 20)
        print(f"  {mt:12s}: {r['accuracy']:.4f}  {bar}")
    best_mt = max(results, key=lambda mt: results[mt]["accuracy"])
    print(f"\n  최고 성능: [{best_mt}]  {results[best_mt]['accuracy']:.4f}")
    print(f"  비교 차트: {run_dir / 'model_comparison.png'}")
    print(f"{'='*60}\n")

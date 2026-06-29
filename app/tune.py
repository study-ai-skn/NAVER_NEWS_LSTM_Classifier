"""Optuna 기반 하이퍼파라미터 자동 튜닝 모듈.

BBC train.py 대비 개선 사항:
  - Optuna 베이지안 서치: LSTM / KoBERT / KoELECTRA 세 모델 유형 동시 탐색
  - 조건부(conditional) 하이퍼파라미터: 모델 유형별로 탐색 공간 분기
  - MedianPruner 로 성능이 낮은 trial 조기 종료
  - fANOVA 파라미터 중요도 + 최적화 이력 시각화 저장
  - SQLite DB 로 study 결과 영구 저장
  - KoNLPy 형태소 분석 1회 수행 후 모든 LSTM trial 에서 공유
  - 트랜스포머 토크나이저는 캐시로 공유 (모델당 최초 1회만 다운로드)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import optuna
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier, TransformerClassifier, MODEL_HF_NAME
from app.preprocess import (
    build_vocab, clean_text, encode_labels, pad_sequences, texts_to_sequences,
    get_transformer_tokenizer, tokenize_for_transformer,
)
from app.train import _evaluate, _train_epoch, set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunsl.ttf",
    "C:/Users/playdata2/Documents/llm_workspace/konlpy_practice_project/fonts/malgunsl.ttf",
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


# ── LSTM 탐색 공간 ─────────────────────────────────────────────────────────────
_LSTM_SPACE = {
    "max_vocab"    : [3000, 5000, 8000, 10000],
    "max_len"      : [10, 15, 20, 25, 30],
    "embed_dim"    : [64, 128, 256],
    "hidden_dim"   : [64, 128, 256],
    "num_layers"   : (1, 3),
    "bidirectional": [True, False],
    "optimizer"    : ["Adam", "AdamW"],
    "weight_decay" : (1e-5, 1e-2),
    "lr"           : (1e-4, 5e-3),  # 상한 5e-3: 클래스 붕괴 방지
    "batch_size"   : [16, 32, 64],
}

# ── 트랜스포머 탐색 공간 ───────────────────────────────────────────────────────
_TR_SPACE = {
    "max_len"      : [64, 128],        # 긴 시퀀스는 CPU 에서 너무 느림
    "weight_decay" : (1e-4, 1e-2),
    "lr"           : (1e-5, 5e-5),    # 파인튜닝 전용 소 LR 범위
    "batch_size"   : [8, 16],
}


TUNE_EPOCHS_LSTM = 15
TUNE_EPOCHS_TR   = 5   # 트랜스포머는 사전학습 덕에 빠르게 수렴
PATIENCE         = 3


def objective(
    trial: optuna.Trial,
    raw_texts: List[str],
    cleaned_texts: List[str],
    labels: List[str],
    config_base: Config,
    tokenizer_cache: dict,
) -> float:
    """Optuna 목적 함수 — 검증 정확도를 반환한다 (방향: maximize).

    model_type 에 따라 LSTM / KoBERT / KoELECTRA 로 분기해
    각 모델에 맞는 하이퍼파라미터를 탐색한다.
    """
    # ── 공통 파라미터 ─────────────────────────────────────────────────────────
    model_type = trial.suggest_categorical("model_type", ["LSTM", "KoBERT", "KoELECTRA"])
    dropout    = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)

    y, label_to_id, _ = encode_labels(labels)
    n_classes          = len(label_to_id)

    # ── 라벨 분할 (테스트셋 제거 후 train/val 만 사용) ────────────────────────
    all_idx = np.arange(len(raw_texts))
    tv_idx, _ = train_test_split(
        all_idx, test_size=config_base.test_size,
        stratify=y, random_state=config_base.random_state,
    )
    val_n     = max(1, int(len(tv_idx) * config_base.val_size))
    val_idx   = tv_idx[:val_n]
    train_idx = tv_idx[val_n:]

    set_seed(42 + trial.number)

    # ── LSTM 분기 ─────────────────────────────────────────────────────────────
    if model_type == "LSTM":
        max_vocab    = trial.suggest_categorical("lstm_max_vocab",     _LSTM_SPACE["max_vocab"])
        max_len      = trial.suggest_categorical("lstm_max_len",       _LSTM_SPACE["max_len"])
        embed_dim    = trial.suggest_categorical("lstm_embed_dim",     _LSTM_SPACE["embed_dim"])
        hidden_dim   = trial.suggest_categorical("lstm_hidden_dim",    _LSTM_SPACE["hidden_dim"])
        num_layers   = trial.suggest_int("lstm_num_layers",            *_LSTM_SPACE["num_layers"])
        bidir        = trial.suggest_categorical("lstm_bidirectional", _LSTM_SPACE["bidirectional"])
        opt_name     = trial.suggest_categorical("lstm_optimizer",     _LSTM_SPACE["optimizer"])
        weight_decay = trial.suggest_float("lstm_weight_decay",        *_LSTM_SPACE["weight_decay"], log=True)
        lr           = trial.suggest_float("lstm_lr",                  *_LSTM_SPACE["lr"],           log=True)
        batch_size   = trial.suggest_categorical("lstm_batch_size",    _LSTM_SPACE["batch_size"])

        vocab     = build_vocab(cleaned_texts, max_vocab)
        sequences = texts_to_sequences(cleaned_texts, vocab)
        x         = pad_sequences(sequences, max_len)

        x_train = x[train_idx]; x_val = x[val_idx]
        y_train = y[train_idx]; y_val = y[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
            batch_size=batch_size,
        )

        model = TextLSTMClassifier(
            vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_classes=n_classes, num_layers=num_layers,
            bidirectional=bidir, dropout=dropout,
        )
        opt_cls   = torch.optim.AdamW if opt_name == "AdamW" else torch.optim.Adam
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        is_tr     = False
        tune_epochs = TUNE_EPOCHS_LSTM

    # ── KoBERT / KoELECTRA 분기 ───────────────────────────────────────────────
    else:
        max_len      = trial.suggest_categorical("tr_max_len",      _TR_SPACE["max_len"])
        weight_decay = trial.suggest_float("tr_weight_decay",       *_TR_SPACE["weight_decay"], log=True)
        lr           = trial.suggest_float("tr_lr",                 *_TR_SPACE["lr"],           log=True)
        batch_size   = trial.suggest_categorical("tr_batch_size",   _TR_SPACE["batch_size"])

        hf_name = MODEL_HF_NAME[model_type]
        if hf_name not in tokenizer_cache:
            print(f"  [{model_type}] 토크나이저 로드: {hf_name}")
            tokenizer_cache[hf_name] = get_transformer_tokenizer(hf_name)
        tokenizer                     = tokenizer_cache[hf_name]
        input_ids, attention_mask     = tokenize_for_transformer(raw_texts, tokenizer, max_len)

        train_loader = DataLoader(
            TensorDataset(input_ids[train_idx], attention_mask[train_idx], torch.tensor(y[train_idx])),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(input_ids[val_idx], attention_mask[val_idx], torch.tensor(y[val_idx])),
            batch_size=batch_size,
        )

        model     = TransformerClassifier(hf_name, num_classes=n_classes, dropout=dropout)
        no_decay  = ["bias", "LayerNorm.weight"]
        wd        = weight_decay if weight_decay > 0 else 0.01
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": wd},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
        criterion = nn.CrossEntropyLoss()
        is_tr     = True
        tune_epochs = TUNE_EPOCHS_TR

    # ── 학습 ──────────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(tune_epochs):
        _train_epoch(model, train_loader, criterion, optimizer, is_transformer=is_tr)
        _, val_acc = _evaluate(model, val_loader, criterion, is_transformer=is_tr)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    return best_val_acc


def _plot_tuning_results(study: optuna.Study, save_dir: str) -> None:
    """최적화 이력 + 모델 유형별 성능 분포 + 파라미터 중요도를 저장한다."""
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        return

    values      = [t.value for t in completed]
    best_so_far = np.maximum.accumulate(values)
    model_types = [t.params.get("model_type", "?") for t in completed]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # ── 최적화 이력 ──────────────────────────────────────────────────────────
    color_map = {"LSTM": "steelblue", "KoBERT": "tomato", "KoELECTRA": "mediumseagreen"}
    colors    = [color_map.get(m, "gray") for m in model_types]
    axes[0].scatter(range(1, len(values) + 1), values, c=colors, alpha=0.7, zorder=3, label=None)
    axes[0].plot(range(1, len(best_so_far) + 1), best_so_far, "k-", linewidth=2.5, label="Best so far")
    for mt, col in color_map.items():
        axes[0].scatter([], [], c=col, label=mt)
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Optuna 최적화 이력 (모델 유형별 색상)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # ── 모델 유형별 평균 정확도 ───────────────────────────────────────────────
    from collections import defaultdict
    per_model: Dict[str, List[float]] = defaultdict(list)
    for t, mt in zip(completed, model_types):
        per_model[mt].append(t.value)
    mt_names  = sorted(per_model.keys())
    mt_means  = [np.mean(per_model[m]) for m in mt_names]
    mt_maxes  = [np.max(per_model[m])  for m in mt_names]
    x_pos     = range(len(mt_names))
    bars = axes[1].bar(x_pos, mt_means, color=[color_map.get(m, "gray") for m in mt_names], alpha=0.8, label="평균")
    axes[1].scatter(x_pos, mt_maxes, color="gold", edgecolors="black", zorder=5, s=100, label="최고")
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(mt_names)
    axes[1].set_ylabel("Validation Accuracy"); axes[1].set_title("모델 유형별 성능 비교")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, mt_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, mean + 0.005, f"{mean:.3f}",
                     ha="center", va="bottom", fontsize=9)

    # ── 파라미터 중요도 (fANOVA) ─────────────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        names  = list(importances.keys())[:10]   # 상위 10개
        scores = [importances[n] for n in names]
        c      = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(names)))
        axes[2].barh(names, scores, color=c)
        axes[2].set_xlabel("중요도 (fANOVA)"); axes[2].set_title("파라미터 중요도")
        axes[2].invert_yaxis()
        for i, s in enumerate(scores):
            axes[2].text(s + 0.002, i, f"{s:.3f}", va="center", fontsize=9)
    except Exception:
        axes[2].axis("off")
        bp = study.best_params
        table = axes[2].table(
            cellText=[[k, str(round(v, 6) if isinstance(v, float) else v)] for k, v in list(bp.items())[:12]],
            colLabels=["파라미터", "최적값"], loc="center", cellLoc="center",
        )
        table.auto_set_font_size(True); table.scale(1, 1.8)
        axes[2].set_title("최적 하이퍼파라미터")

    plt.suptitle(
        f"Optuna 튜닝 결과  (Best val_acc = {study.best_value:.4f}  |  Best model: {study.best_params.get('model_type','?')})",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "optuna_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Optuna 결과 시각화 저장 완료: optuna_results.png")


def _params_to_config(params: dict) -> Config:
    """Optuna trial.params 를 Config 로 변환한다."""
    model_type = params["model_type"]
    if model_type == "LSTM":
        return Config(
            model_type     = model_type,
            max_vocab      = params.get("lstm_max_vocab",     5000),
            max_len        = params.get("lstm_max_len",        20),
            embed_dim      = params.get("lstm_embed_dim",     128),
            hidden_dim     = params.get("lstm_hidden_dim",    128),
            num_layers     = params.get("lstm_num_layers",      2),
            bidirectional  = params.get("lstm_bidirectional", True),
            dropout        = params.get("dropout",            0.3),
            optimizer_name = params.get("lstm_optimizer",   "Adam"),
            weight_decay   = params.get("lstm_weight_decay", 0.0),
            learning_rate  = params.get("lstm_lr",          0.001),
            batch_size     = params.get("lstm_batch_size",    16),
        )
    else:
        return Config(
            model_type    = model_type,
            max_len       = params.get("tr_max_len",      128),
            dropout       = params.get("dropout",         0.1),
            optimizer_name= "AdamW",
            weight_decay  = params.get("tr_weight_decay", 0.01),
            learning_rate = params.get("tr_lr",           2e-5),
            batch_size    = params.get("tr_batch_size",    16),
        )


def run_tuning(n_trials: int = 30, save_dir: str | None = None) -> Dict[str, Config]:
    """Optuna 하이퍼파라미터 튜닝을 실행하고 최적 Config 를 반환한다.

    LSTM / KoBERT / KoELECTRA 세 모델 유형을 함께 탐색한다.
    KoNLPy 형태소 분석은 한 번만 수행하고 LSTM trial 에서 공유한다.
    트랜스포머 토크나이저는 캐시로 공유한다.
    """
    print(f"\n{'='*60}")
    print(f"  Optuna 하이퍼파라미터 튜닝 시작  (n_trials={n_trials})")
    print(f"  탐색 모델: LSTM / KoBERT / KoELECTRA")
    print(f"{'='*60}\n")

    config_base   = Config()
    raw_texts, labels = load_sample_data()

    print("KoNLPy 형태소 분석 중 (LSTM trial 용, 최초 1회)...")
    cleaned_texts = [clean_text(t) for t in raw_texts]
    print(f"전처리 완료: {len(cleaned_texts)}건\n")

    if save_dir is None:
        save_dir = os.path.dirname(config_base.model_path)
    os.makedirs(save_dir, exist_ok=True)
    db_path = os.path.join(save_dir, "optuna_study.db")

    tokenizer_cache: dict = {}

    try:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            study_name="naver_news_multimodel",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
        )
    except Exception:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )

    study.optimize(
        lambda trial: objective(trial, raw_texts, cleaned_texts, labels, config_base, tokenizer_cache),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(
                f"  Trial {trial.number:>3d} [{trial.params.get('model_type','?'):10s}] "
                f"val_acc={trial.value:.4f} | best={study.best_value:.4f}"
            ) if trial.value is not None else None
        ],
    )

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    bp         = study.best_params
    best_model = bp["model_type"]
    print(f"\n{'='*60}")
    print(f"  최적 모델 유형: {best_model}")
    print(f"  최고 검증 정확도: {study.best_value:.4f}")
    print(f"{'='*60}")
    for k, v in bp.items():
        print(f"  {k:25s}: {str(v)}")
    completed = [t for t in study.trials if t.value is not None]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n  완료 trials: {len(completed)}  |  Pruned: {len(pruned)}")
    print(f"{'='*60}\n")

    _plot_tuning_results(study, save_dir)

    # ── 모델 유형별 최적 Config 추출 ──────────────────────────────────────────
    completed    = [t for t in study.trials if t.value is not None]
    best_configs: Dict[str, Config] = {}
    for mt in ["LSTM", "KoBERT", "KoELECTRA"]:
        type_trials = [t for t in completed if t.params.get("model_type") == mt]
        if not type_trials:
            continue
        best_trial       = max(type_trials, key=lambda t: t.value)
        cfg              = _params_to_config(best_trial.params)
        best_configs[mt] = cfg
        print(f"  [{mt}] 최고 val_acc = {best_trial.value:.4f}  →  {best_trial.params}")

    # ── best_configs.json 저장 (main.py 가 읽어서 학습에 사용) ───────────────
    json_path = os.path.join(save_dir, "best_configs.json")
    json_data: dict = {}
    for mt, cfg in best_configs.items():
        d = dataclasses.asdict(cfg)
        d.pop("model_path", None)        # 경로는 실행 시 결정하므로 제외
        json_data[mt] = d
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n모델별 최적 Config 저장 완료: {json_path}")
    print("  → main.py 를 실행하면 세 모델을 순서대로 학습합니다.\n")

    return best_configs

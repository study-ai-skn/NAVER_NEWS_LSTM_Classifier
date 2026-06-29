"""데이터 전처리, 모델 학습, 평가, 저장을 수행하는 모듈.

BBC train.py와 동일한 함수명(set_seed, train_model)을 유지하면서
다음 항목을 개선했다 (Reuters_News_classification_improved.ipynb 참고):
  - 훈련셋에서 검증셋 분리 (데이터 누수 방지)
  - ReduceLROnPlateau 학습률 스케줄러 (LSTM)
  - Linear Warmup 스케줄러 (KoBERT / KoELECTRA)
  - Gradient Clipping (max_norm=1.0)
  - Early Stopping
  - 모델 유형 선택: LSTM / KoBERT / KoELECTRA
  - 전처리 단어 빈도 시각화: 워드클라우드 + 막대 차트
  - 학습 곡선 시각화 (정확도/손실/학습률)
  - 혼동 행렬 시각화
"""

from __future__ import annotations

import os
import pickle
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from app.config import Config
from app.data import load_sample_data
from app.model import TextLSTMClassifier, TransformerClassifier, MODEL_HF_NAME
from app.preprocess import (
    build_vocab, clean_text, encode_labels, pad_sequences, texts_to_sequences,
    precompute_tokens, build_vocab_from_tokens, tokens_to_sequences,
    get_transformer_tokenizer, tokenize_for_transformer,
)


# ── 한국어 폰트 자동 탐색 ──────────────────────────────────────────────────────
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


def set_seed(seed: int) -> None:
    """학습 결과가 최대한 동일하게 재현되도록 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    is_transformer: bool = False,
    warmup_scheduler=None,
) -> Tuple[float, float]:
    """한 에포크 학습을 수행하고 평균 손실과 정확도를 반환한다."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        if is_transformer:
            input_ids, attention_mask, batch_y = batch
            logits = model(input_ids, attention_mask)
        else:
            batch_x, batch_y = batch
            logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        total_loss += loss.item() * batch_y.size(0)
        correct    += (torch.argmax(logits, dim=1) == batch_y).sum().item()
        total      += batch_y.size(0)
    return total_loss / total, correct / total


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    is_transformer: bool = False,
) -> Tuple[float, float]:
    """검증/테스트 데이터로 손실과 정확도를 계산해 반환한다."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if is_transformer:
                input_ids, attention_mask, batch_y = batch
                logits = model(input_ids, attention_mask)
            else:
                batch_x, batch_y = batch
                logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_y.size(0)
            correct    += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            total      += batch_y.size(0)
    return total_loss / total, correct / total


def _plot_word_freq(word_freq: Dict[str, int], save_dir: str, top_n: int = 20) -> None:
    """전처리된 단어 빈도를 워드클라우드 + 막대 차트로 시각화한다."""
    if not word_freq:
        return

    top_items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top_items:
        return
    words, freqs = zip(*top_items)
    top_dict = dict(top_items)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    try:
        from wordcloud import WordCloud
        wc_kwargs: dict = dict(width=600, height=420, background_color="white", max_words=top_n)
        if _KOREAN_FONT_PATH:
            wc_kwargs["font_path"] = _KOREAN_FONT_PATH
        wc = WordCloud(**wc_kwargs).generate_from_frequencies(top_dict)
        axes[0].imshow(wc, interpolation="bilinear")
        axes[0].axis("off")
        axes[0].set_title(f"워드클라우드  (Top {top_n})", fontsize=14)
    except ImportError:
        axes[0].text(0.5, 0.5, "wordcloud 패키지 미설치\npip install wordcloud",
                     ha="center", va="center", transform=axes[0].transAxes, fontsize=11)
        axes[0].set_title("워드클라우드 (wordcloud 패키지 필요)")

    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(words)))
    axes[1].barh(range(len(words)), freqs, color=colors)
    axes[1].set_yticks(range(len(words)))
    axes[1].set_yticklabels(words, fontsize=10)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("빈도")
    axes[1].set_title(f"상위 {top_n}개 단어 빈도", fontsize=14)
    for i, freq in enumerate(freqs):
        axes[1].text(freq + 0.1, i, str(freq), va="center", fontsize=9)

    plt.suptitle("전처리 단어 분석", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "word_frequency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("단어 빈도 시각화 저장 완료: word_frequency.png")


def _plot_history(train_losses: List[float], val_losses: List[float],
                  train_accs: List[float],  val_accs: List[float],
                  lr_history: List[float],  save_dir: str) -> None:
    """학습 곡선(정확도·손실·학습률)을 파일로 저장한다."""
    epochs_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, train_accs, label="Train Accuracy")
    axes[0].plot(epochs_range, val_accs,   label="Validation Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend()

    axes[1].plot(epochs_range, train_losses, label="Train Loss")
    axes[1].plot(epochs_range, val_losses,   label="Validation Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss"); axes[1].legend()

    axes[2].plot(epochs_range, lr_history, label="Learning Rate", color="green")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("학습 곡선 저장 완료: training_curves.png")


def _plot_confusion_matrix(all_targets: List[int], all_preds: List[int],
                           id_to_label: Dict[int, str], save_dir: str) -> None:
    """혼동 행렬을 파일로 저장한다."""
    labels  = [id_to_label[i] for i in range(len(id_to_label))]
    cm      = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im      = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("예측 라벨"); ax.set_ylabel("실제 라벨")
    ax.set_title("혼동 행렬 (Confusion Matrix)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("혼동 행렬 저장 완료: confusion_matrix.png")


def train_model(config: Config):
    """네이버 뉴스 데이터로 분류 모델을 학습한다.

    config.model_type 에 따라 LSTM / KoBERT / KoELECTRA 중 하나를 선택한다.
    """
    set_seed(config.random_state)
    raw_texts, labels = load_sample_data(max_items=getattr(config, "max_items", 300))
    y, label_to_id, id_to_label = encode_labels(labels)

    save_dir = os.path.dirname(config.model_path)
    os.makedirs(save_dir, exist_ok=True)

    is_transformer = config.model_type in ("KoBERT", "KoELECTRA")

    # ── 전처리 & 데이터셋 구성 ─────────────────────────────────────────────────
    if is_transformer:
        hf_name   = MODEL_HF_NAME[config.model_type]
        print(f"[{config.model_type}] 토크나이저 로드 중: {hf_name}")
        tokenizer = get_transformer_tokenizer(hf_name)
        input_ids, attention_mask = tokenize_for_transformer(raw_texts, tokenizer, config.max_len)

        # 인덱스 기반 분할
        all_idx = np.arange(len(raw_texts))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=config.test_size, stratify=y, random_state=config.random_state
        )
        val_n     = max(1, int(len(train_val_idx) * config.val_size))
        val_idx   = train_val_idx[:val_n]
        train_idx = train_val_idx[val_n:]

        y_t = torch.tensor(y)
        train_ds = TensorDataset(input_ids[train_idx], attention_mask[train_idx], y_t[train_idx])
        val_ds   = TensorDataset(input_ids[val_idx],   attention_mask[val_idx],   y_t[val_idx])
        test_ds  = TensorDataset(input_ids[test_idx],  attention_mask[test_idx],  y_t[test_idx])

        # 단어 빈도 (간이 버전: KoNLPy 없이 공백 분리)
        wf: Counter = Counter()
        for t in raw_texts:
            for w in re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", t).split():
                if len(w) >= 2:
                    wf[w] += 1
        word_freq = dict(wf)

    else:
        cleaned_texts = [clean_text(t) for t in raw_texts]
        use_morph     = getattr(config, "use_morphemes", False)
        token_lists   = precompute_tokens(cleaned_texts, use_morphemes=use_morph)
        vocab         = build_vocab_from_tokens(token_lists, config.max_vocab)
        sequences     = tokens_to_sequences(token_lists, vocab)
        x             = pad_sequences(sequences, config.max_len)

        id_to_word = {v: k for k, v in vocab.items() if k not in ("<PAD>", "<UNK>")}
        word_freq: Dict[str, int] = {}
        for seq in sequences:
            for wid in seq:
                if wid in id_to_word:
                    w = id_to_word[wid]
                    word_freq[w] = word_freq.get(w, 0) + 1

        x_train_all, x_test, y_train_all, y_test = train_test_split(
            x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )
        val_count   = max(1, int(len(x_train_all) * config.val_size))
        train_count = len(x_train_all) - val_count
        full_train_ds = TensorDataset(torch.tensor(x_train_all), torch.tensor(y_train_all))
        test_ds       = TensorDataset(torch.tensor(x_test),      torch.tensor(y_test))
        train_ds, val_ds = random_split(
            full_train_ds, [train_count, val_count],
            generator=torch.Generator().manual_seed(config.random_state)
        )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size)

    n_train = len(train_ds); n_val = len(val_ds); n_test = len(test_ds)
    print(f"훈련 샘플: {n_train}  |  검증 샘플: {n_val}  |  테스트 샘플: {n_test}")
    print(f"카테고리: {list(label_to_id.keys())}")

    _plot_word_freq(word_freq, save_dir)

    # ── 모델 & 옵티마이저 ──────────────────────────────────────────────────────
    if is_transformer:
        model     = TransformerClassifier(hf_name, num_classes=len(label_to_id), dropout=config.dropout)
        no_decay  = ["bias", "LayerNorm.weight"]
        wd        = config.weight_decay if config.weight_decay > 0 else 0.01
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": wd},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)
        # Linear warmup (첫 10% step)
        from transformers import get_linear_schedule_with_warmup
        total_steps  = config.epochs * len(train_loader)
        warmup_steps = max(1, total_steps // 10)
        warmup_sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        plateau_sched = None
    else:
        model     = TextLSTMClassifier(
            vocab_size=len(vocab), embed_dim=config.embed_dim, hidden_dim=config.hidden_dim,
            num_classes=len(label_to_id), num_layers=config.num_layers,
            bidirectional=config.bidirectional, dropout=config.dropout,
        )
        opt_cls   = torch.optim.AdamW if config.optimizer_name == "AdamW" else torch.optim.Adam
        optimizer = opt_cls(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
        warmup_sched  = None

    criterion = nn.CrossEntropyLoss()

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    # LSTM: val_acc 기준 (손실보다 정확도가 일반화 지표로 더 적합)
    # Transformer: val_loss 기준 (표준 파인튜닝 관행)
    use_acc_criterion = not is_transformer

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    lr_history               = []
    best_val_metric   = 0.0 if use_acc_criterion else float("inf")
    epochs_no_improve = 0

    print(f"\n[{config.model_type}] 학습 시작  (epochs={config.epochs}, early_stop={'val_acc' if use_acc_criterion else 'val_loss'})\n")
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer,
            is_transformer=is_transformer, warmup_scheduler=warmup_sched,
        )
        val_loss, val_acc = _evaluate(model, val_loader, criterion, is_transformer=is_transformer)

        if plateau_sched is not None:
            plateau_sched.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc)
        lr_history.append(current_lr)

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"| train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} "
            f"| val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} "
            f"| lr: {current_lr:.2e}"
        )

        # 개선 여부 판단 (LSTM=acc 최대화, Transformer=loss 최소화)
        improved = (val_acc > best_val_metric) if use_acc_criterion else (val_loss < best_val_metric)
        if improved:
            best_val_metric   = val_acc if use_acc_criterion else val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                metric_name = f"val_acc={best_val_metric:.4f}" if use_acc_criterion else f"val_loss={best_val_metric:.4f}"
                print(f"\nEarly Stopping: {epoch} 에포크에서 조기 종료 (best {metric_name})")
                break

    # ── 최적 모델 로드 & 최종 평가 ────────────────────────────────────────────
    if is_transformer:
        model = TransformerClassifier(hf_name, num_classes=len(label_to_id), dropout=config.dropout)
    else:
        model = TextLSTMClassifier(
            vocab_size=len(vocab), embed_dim=config.embed_dim, hidden_dim=config.hidden_dim,
            num_classes=len(label_to_id), num_layers=config.num_layers,
            bidirectional=config.bidirectional, dropout=config.dropout,
        )
    model.load_state_dict(torch.load(config.model_path, map_location="cpu", weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if is_transformer:
                input_ids, attention_mask, batch_y = batch
                logits = model(input_ids, attention_mask)
            else:
                batch_x, batch_y = batch
                logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n테스트 정확도: {accuracy:.4f}  [{config.model_type}]")
    print(classification_report(
        all_targets, all_preds,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        zero_division=0,
    ))

    _plot_history(train_losses, val_losses, train_accs, val_accs, lr_history, save_dir)
    _plot_confusion_matrix(all_targets, all_preds, id_to_label, save_dir)

    meta = {
        "vocab":        vocab if not is_transformer else {},
        "label_to_id":  label_to_id,
        "id_to_label":  id_to_label,
        "config":       config,
    }
    meta_path = config.model_path.replace(".pt", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    return model, {**meta, "accuracy": accuracy}

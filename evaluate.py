"""기학습 모델을 로드해 정확도와 분류 리포트만 출력하는 스크립트.

별도 학습 없이 이미 저장된 모델을 불러와 현재 데이터로 평가한다.

사용법:
  python evaluate.py                         # 최신 버전의 모든 모델 평가
  python evaluate.py --version v4            # v4 폴더의 모든 모델 평가
  python evaluate.py --version v4 --model LSTM        # 특정 모델만 평가
  python evaluate.py --text "뉴스 제목 입력"             # 단일 뉴스 예측
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.config import Config
from app.data import load_sample_data
from app.model import MODEL_HF_NAME, TextLSTMClassifier, TransformerClassifier
from app.predict import load_artifacts, predict_text
from app.preprocess import (
    clean_text, encode_labels, pad_sequences, texts_to_sequences,
    get_transformer_tokenizer, tokenize_for_transformer,
)

_MODELS_DIR = Path(__file__).parent / "models"

# 모델 파일명 패턴: naver_{model_type_lower}_model.pt
_MODEL_FILE_MAP = {
    "LSTM":      "naver_lstm_model.pt",
    "KoBERT":    "naver_kobert_model.pt",
    "KoELECTRA": "naver_koelectra_model.pt",
}


def _latest_version() -> str | None:
    """가장 최신 models/v{N} 폴더 이름을 반환한다."""
    if not _MODELS_DIR.exists():
        return None
    versions = sorted(
        int(d.name[1:])
        for d in _MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^v\d+$", d.name)
    )
    return f"v{versions[-1]}" if versions else None


def _find_models(version_dir: Path) -> Dict[str, Path]:
    """버전 폴더에서 존재하는 모델 파일 경로를 반환한다."""
    found = {}
    for mt, fname in _MODEL_FILE_MAP.items():
        p = version_dir / fname
        if p.exists():
            found[mt] = p
    return found


def evaluate_model(model_path: Path) -> None:
    """단일 모델 파일을 로드해 테스트셋 정확도를 출력한다."""
    meta_path = Path(str(model_path).replace(".pt", "_meta.pkl"))
    if not meta_path.exists():
        print(f"  [오류] 메타 파일 없음: {meta_path}")
        return

    import pickle
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    saved_cfg: Config = metadata["config"]
    label_to_id: dict = metadata["label_to_id"]
    id_to_label: dict = metadata["id_to_label"]
    is_transformer = saved_cfg.model_type in ("KoBERT", "KoELECTRA")

    print(f"\n[{saved_cfg.model_type}]  {model_path}")

    # ── 데이터 준비 (학습 때와 동일한 split) ───────────────────────────────────
    raw_texts, labels = load_sample_data()
    y, _, _ = encode_labels(labels)

    all_idx = np.arange(len(raw_texts))
    _, test_idx = train_test_split(
        all_idx,
        test_size=saved_cfg.test_size,
        stratify=y,
        random_state=saved_cfg.random_state,
    )

    if is_transformer:
        hf_name  = MODEL_HF_NAME[saved_cfg.model_type]
        tokenizer = get_transformer_tokenizer(hf_name)
        input_ids, attention_mask = tokenize_for_transformer(raw_texts, tokenizer, saved_cfg.max_len)
        test_ds = TensorDataset(input_ids[test_idx], attention_mask[test_idx], torch.tensor(y[test_idx]))
        model   = TransformerClassifier(hf_name, num_classes=len(label_to_id), dropout=saved_cfg.dropout)
    else:
        vocab     = metadata.get("vocab", {})
        cleaned   = [clean_text(t) for t in raw_texts]
        sequences = texts_to_sequences(cleaned, vocab)
        x         = pad_sequences(sequences, saved_cfg.max_len)
        test_ds   = TensorDataset(torch.tensor(x[test_idx]), torch.tensor(y[test_idx]))
        model     = TextLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=saved_cfg.embed_dim,
            hidden_dim=saved_cfg.hidden_dim,
            num_classes=len(label_to_id),
            num_layers=saved_cfg.num_layers,
            bidirectional=saved_cfg.bidirectional,
            dropout=saved_cfg.dropout,
        )

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(test_ds, batch_size=32)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_transformer:
                ids, mask, batch_y = batch
                logits = model(ids, mask)
            else:
                batch_x, batch_y = batch
                logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch_y.tolist())

    acc = accuracy_score(all_targets, all_preds)
    print(f"  테스트 정확도: {acc:.4f}  ({int(acc * len(all_targets))}/{len(all_targets)})")
    print(classification_report(
        all_targets, all_preds,
        target_names=[id_to_label[i] for i in range(len(id_to_label))],
        zero_division=0,
    ))


def predict_single(text: str, version_dir: Path) -> None:
    """단일 뉴스 제목을 버전 폴더의 모든 학습된 모델로 예측한다."""
    found = _find_models(version_dir)
    if not found:
        print(f"학습된 모델 없음: {version_dir}")
        return
    print(f"\n뉴스: {text}\n")
    for mt, model_path in found.items():
        meta_path = Path(str(model_path).replace(".pt", "_meta.pkl"))
        if not meta_path.exists():
            continue
        import pickle
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        saved_cfg = metadata["config"]
        # 임시로 Config.model_path 를 실제 경로로 교체해 load_artifacts 호출
        saved_cfg.model_path = str(model_path)
        model, meta = load_artifacts(saved_cfg)
        predict_text(text, model, meta, saved_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="기학습 모델 평가 스크립트")
    parser.add_argument("--version", type=str, default=None,
                        help="평가할 버전 폴더 (예: v4). 미지정 시 최신 버전 사용")
    parser.add_argument("--model",   type=str, default=None,
                        choices=["LSTM", "KoBERT", "KoELECTRA"],
                        help="평가할 모델 유형. 미지정 시 폴더 내 모든 모델 평가")
    parser.add_argument("--text",    type=str, default=None,
                        help="단일 뉴스 제목 예측 모드")
    args = parser.parse_args()

    version = args.version or _latest_version()
    if version is None:
        print("오류: models/ 폴더에 학습된 모델이 없습니다.")
        return

    version_dir = _MODELS_DIR / version
    if not version_dir.exists():
        print(f"오류: {version_dir} 폴더가 없습니다.")
        return

    print(f"\n평가 버전: {version}  ({version_dir})")

    # ── 단일 뉴스 예측 모드 ───────────────────────────────────────────────────
    if args.text:
        predict_single(args.text, version_dir)
        return

    # ── 정확도 평가 모드 ──────────────────────────────────────────────────────
    found = _find_models(version_dir)
    if not found:
        print(f"학습된 모델 파일 없음: {version_dir}")
        return

    target_models = {args.model: found[args.model]} if (args.model and args.model in found) else found
    for mt, model_path in target_models.items():
        evaluate_model(model_path)


if __name__ == "__main__":
    main()

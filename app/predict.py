from __future__ import annotations

import pickle
from typing import Dict, Tuple

import torch

from app.config import Config
from app.model import TextLSTMClassifier, TransformerClassifier, MODEL_HF_NAME
from app.preprocess import (
    clean_text, pad_sequences, texts_to_sequences,
    get_transformer_tokenizer, tokenize_for_transformer,
)


def load_artifacts(config: Config):
    """저장된 모델과 메타데이터를 불러온다. model_type 에 따라 모델 클래스를 선택한다."""
    meta_path = config.model_path.replace(".pt", "_meta.pkl")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    saved_cfg = metadata["config"]

    if saved_cfg.model_type in ("KoBERT", "KoELECTRA"):
        hf_name = MODEL_HF_NAME[saved_cfg.model_type]
        model   = TransformerClassifier(
            model_name=hf_name,
            num_classes=len(metadata["label_to_id"]),
            dropout=saved_cfg.dropout,
        )
    else:
        model = TextLSTMClassifier(
            vocab_size=len(metadata["vocab"]),
            embed_dim=saved_cfg.embed_dim,
            hidden_dim=saved_cfg.hidden_dim,
            num_classes=len(metadata["label_to_id"]),
            num_layers=saved_cfg.num_layers,
            bidirectional=saved_cfg.bidirectional,
            dropout=saved_cfg.dropout,
        )

    model.load_state_dict(torch.load(config.model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model, metadata


def predict_text(text: str, model, metadata: Dict[str, object], config: Config) -> str:
    """뉴스 제목 하나의 카테고리를 예측해 반환한다."""
    saved_cfg = metadata["config"]

    if saved_cfg.model_type in ("KoBERT", "KoELECTRA"):
        hf_name   = MODEL_HF_NAME[saved_cfg.model_type]
        tokenizer = get_transformer_tokenizer(hf_name)
        input_ids, attention_mask = tokenize_for_transformer([text], tokenizer, saved_cfg.max_len)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
    else:
        cleaned  = clean_text(text)
        sequence = texts_to_sequences([cleaned], metadata["vocab"])
        padded   = pad_sequences(sequence, saved_cfg.max_len)
        x        = torch.tensor(padded, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits = model(x)

    probs     = torch.softmax(logits, dim=1)
    pred_id   = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(torch.max(probs).item())
    label     = metadata["id_to_label"][pred_id]
    print(f"Predicted category: {label}  (probability: {pred_prob:.4f})  [{saved_cfg.model_type}]")
    return label


if __name__ == "__main__":
    config = Config()
    model, metadata = load_artifacts(config)
    sample_news = "삼성전자 반도체 기술 개발 성공 발표"
    print("\nNews:", sample_news)
    predict_text(sample_news, model, metadata, config)

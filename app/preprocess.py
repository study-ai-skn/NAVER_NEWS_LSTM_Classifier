"""한국어 뉴스 텍스트 전처리 모듈.

BBC preprocess.py와 동일한 함수명(build_vocab, clean_text, encode_labels,
pad_sequences, texts_to_sequences)을 유지하면서 한국어 처리를 위해 수정했다.
KoNLPy Okt 형태소 분석기를 사용해 명사를 추출한다.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# KoNLPy 로드 전 JAVA_HOME 설정 (konlpy_practice_project 참고)
if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk-21.0.2"

try:
    from konlpy.tag import Okt
    _okt = Okt(max_heap_size=256)
    _USE_KONLPY = True
except Exception:
    _USE_KONLPY = False

# BBC model.py에 정의된 STOP_WORDS를 import해서 전처리에 활용한다.
from app.model import STOP_WORDS


def clean_text(text: str) -> str:
    """한국어 텍스트를 정제한다: 한글과 공백만 남기고 나머지를 제거한다."""
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def _extract_nouns(text: str) -> List[str]:
    """텍스트에서 명사만 추출한다. KoNLPy 사용 불가 시 공백 기준으로 분리한다."""
    if _USE_KONLPY:
        return _okt.nouns(text)
    return [w for w in text.split() if len(w) >= 2]


def _extract_morphemes(text: str) -> List[str]:
    """명사 + 동사 + 형용사 + 부사를 추출한다 (명사만보다 더 많은 문맥 보존).

    KoNLPy 사용 불가 시 공백 분리로 대체한다.
    """
    if _USE_KONLPY:
        keep_pos = {"Noun", "Verb", "Adjective", "Adverb"}
        return [word for word, pos in _okt.pos(text) if pos in keep_pos]
    return [w for w in text.split() if len(w) >= 2]


def precompute_tokens(texts: List[str], use_morphemes: bool = False) -> List[List[str]]:
    """텍스트 목록을 토큰 목록으로 일괄 변환한다 (한 번만 호출해 재사용 권장).

    use_morphemes=False: 명사만 추출
    use_morphemes=True : 명사+동사+형용사+부사 추출
    """
    extractor = _extract_morphemes if use_morphemes else _extract_nouns
    result = []
    for text in texts:
        tokens = [t for t in extractor(text) if t not in STOP_WORDS and len(t) >= 2]
        result.append(tokens if tokens else ["<UNK>"])
    return result


def build_vocab_from_tokens(token_lists: List[List[str]], max_vocab: int) -> Dict[str, int]:
    """사전 추출된 토큰 목록으로 단어 사전을 구성한다."""
    counter: Counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def tokens_to_sequences(token_lists: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    """사전 추출된 토큰 목록을 정수 시퀀스로 변환한다."""
    unk_id = vocab.get("<UNK>", 1)
    sequences = []
    for tokens in token_lists:
        seq = [vocab.get(t, unk_id) for t in tokens]
        sequences.append(seq if seq else [unk_id])
    return sequences


def build_vocab(texts: List[str], max_vocab: int) -> Dict[str, int]:
    """정제된 텍스트 목록에서 단어 사전을 구성한다 (하위 호환 유지)."""
    token_lists = precompute_tokens(texts, use_morphemes=False)
    return build_vocab_from_tokens(token_lists, max_vocab)


def texts_to_sequences(texts: List[str], vocab: Dict[str, int]) -> List[List[int]]:
    """정제된 텍스트 목록을 정수 토큰 시퀀스로 변환한다 (하위 호환 유지)."""
    token_lists = precompute_tokens(texts, use_morphemes=False)
    return tokens_to_sequences(token_lists, vocab)


def pad_sequences(sequences: List[List[int]], max_len: int) -> np.ndarray:
    """모든 시퀀스를 동일한 길이로 패딩한다 (앞쪽에 0을 채운다)."""
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        trunc = seq[-max_len:]                      # 너무 길면 뒷부분을 사용
        padded[i, max_len - len(trunc):] = trunc
    return padded


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """문자열 카테고리 라벨을 정수로 변환한다."""
    unique_labels = sorted(set(labels))
    label_to_id   = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label   = {idx: label for label, idx in label_to_id.items()}
    y = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return y, label_to_id, id_to_label


# ── KoBERT / KoELECTRA 전처리 ─────────────────────────────────────────────────

def get_transformer_tokenizer(model_name: str):
    """HuggingFace AutoTokenizer를 반환한다. 첫 호출 시 모델을 다운로드한다."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_for_transformer(texts: List[str], tokenizer, max_len: int):
    """텍스트 목록을 트랜스포머 입력 텐서(input_ids, attention_mask)로 변환한다."""
    import torch
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]

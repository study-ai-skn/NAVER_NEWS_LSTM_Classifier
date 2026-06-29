import torch
from torch import nn


# BBC 프로젝트와 동일하게 STOP_WORDS를 model.py에 정의한다.
STOP_WORDS = {
    "이", "가", "을", "를", "의", "에", "도", "로", "과", "와", "한", "은", "는",
    "에서", "으로", "에게", "까지", "부터", "서", "대한", "위한", "따른", "통해",
    "위해", "관련", "대해", "하다", "있다", "되다", "수", "것", "그", "및", "등",
    "하여", "통하여", "위하여", "대하여",
}


class TextLSTMClassifier(nn.Module):
    """양방향 다층 LSTM 기반 한국어 뉴스 분류 모델.

    BBC 버전 대비 개선 사항:
    - 양방향(Bidirectional) LSTM: 앞뒤 문맥 동시 학습
    - 2층 LSTM: 더 깊은 특징 추출
    - Global Max Pooling: 전체 시퀀스에서 가장 중요한 특징 추출
    - batch_first=True: 차원 처리 일관성 확보
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3):
        super(TextLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        embedded = self.dropout(self.embedding(x))       # [batch, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embedded)                # [batch, seq_len, hidden*2]
        # Global Max Pooling: 전체 시퀀스에서 가장 중요한 특징을 추출
        pooled = torch.max(lstm_out, dim=1)[0]           # [batch, hidden*2]
        output = self.fc(self.dropout(pooled))           # [batch, num_classes]
        return output


# ── 사전학습 한국어 모델 HuggingFace 이름 매핑 ─────────────────────────────────
MODEL_HF_NAME = {
    "KoBERT":    "klue/bert-base",                           # KLUE BERT (한국어 BERT)
    "KoELECTRA": "monologg/koelectra-base-v3-discriminator", # KoELECTRA v3
}


class TransformerClassifier(nn.Module):
    """HuggingFace 사전학습 한국어 모델 기반 텍스트 분류기.

    [CLS] 토큰 임베딩에 드롭아웃 + 선형 분류 헤드를 붙인 표준 파인튜닝 구조다.
    KoBERT(klue/bert-base) 와 KoELECTRA 모두 동일 코드로 지원한다.
    """

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size   = self.backbone.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0]   # [CLS] 토큰
        return self.fc(self.dropout(cls))

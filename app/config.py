from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Config:
    '''hp 관리용'''

    max_vocab       : int   = 5000
    max_len         : int   = 20        # 한국어 뉴스 제목 기준 명사 최대 개수
    embed_dim       : int   = 128       # BBC 64 → 128 (표현력 향상)
    hidden_dim      : int   = 128       # BBC 64 → 128 (표현력 향상)
    num_layers      : int   = 2         # 양방향 2층 LSTM
    bidirectional   : bool  = True      # 양방향 문맥 학습
    dropout         : float = 0.3       # 과적합 방지
    batch_size      : int   = 16
    epochs          : int   = 30        # Early Stopping으로 자동 종료
    learning_rate   : float = 0.001
    test_size       : float = 0.2
    val_size        : float = 0.2       # 훈련셋에서 검증셋 분리 비율
    random_state    : int   = 42
    patience        : int   = 5         # Early Stopping patience
    optimizer_name  : str   = "Adam"    # "Adam" | "AdamW"
    weight_decay    : float = 0.0       # AdamW weight decay (Adam은 0.0)
    model_type      : str   = "LSTM"    # "LSTM" | "KoBERT" | "KoELECTRA"
    model_path      : str   = str(PROJECT_ROOT / 'models' / 'naver_lstm_model.pt')
    use_morphemes   : bool  = False     # False=명사만, True=명사+동사+형용사 (LSTM 전용)
    max_items       : int   = 300       # 카테고리당 크롤링 최대 건수

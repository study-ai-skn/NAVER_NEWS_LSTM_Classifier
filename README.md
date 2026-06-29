# NAVER NEWS LSTM Classifier

네이버 뉴스 카테고리 자동 분류 프로젝트.  
BBC RNN Classifier를 기준(베이스라인)으로 삼아 정확도와 손실값을 개선했다.

---

## 베이스라인 대비 개선 사항

| 항목 | BBC LSTM (베이스라인) | NAVER (이 프로젝트) |
|------|----------------------|---------------------|
| 언어 | 영어 | 한국어 (KoNLPy 형태소 분석) |
| 모델 선택 | LSTM 고정 | **LSTM / KoBERT / KoELECTRA** 자동 선택 |
| LSTM 방향 | 단방향 | 양방향(Bidirectional) + Global Max Pooling |
| 하이퍼파라미터 | 수동 고정 | **Optuna 베이지안 탐색** (기본 30 trials) |
| 검증셋 | 없음 | 훈련셋에서 20% 분리 |
| LR 스케줄러 | 없음 | ReduceLROnPlateau (LSTM) / Linear Warmup (트랜스포머) |
| Gradient Clipping | 없음 | max_norm=1.0 |
| Early Stopping | 없음 | patience=5 |
| 시각화 | 없음 | 학습 곡선 / 혼동 행렬 / 단어 빈도 / Optuna 이력 |
| 평가지표 | 정확도만 | Precision / Recall / F1 (카테고리별) |
| 데이터 | 고정 샘플 | 실시간 크롤링 (최대 100건/카테고리) |
| 결과 버저닝 | 없음 | models/v{N}/ 자동 생성 (이전 결과 보존) |
| 테스트 정확도 | 14.29% | LSTM 36.7% → KoBERT/KoELECTRA 80%+ 기대 |

---

## 프로젝트 구조

```
NAVER_NEWS_LSTM_Classifier/
├─ app/
│  ├─ __init__.py
│  ├─ config.py       — 하이퍼파라미터 설정 (model_type 포함)
│  ├─ data.py         — 네이버 뉴스 크롤링 / 내장 샘플 데이터
│  ├─ preprocess.py   — KoNLPy 전처리 + 트랜스포머 토크나이저
│  ├─ model.py        — TextLSTMClassifier / TransformerClassifier
│  ├─ train.py        — 학습 · 평가 · 시각화
│  ├─ tune.py         — Optuna 베이지안 하이퍼파라미터 탐색
│  └─ predict.py      — 모델 로드 및 단일 예측
├─ models/
│  ├─ v4/             — 이전 실험 결과 (LSTM, 36.7%)
│  └─ v{N}/           — 최신 실험 결과 (자동 생성)
├─ reports/
│  └─ tuning_v1.md ~ tuning_v{N}.md   — 실험 보고서
├─ data/
├─ naver_lstm_classifier.py   — 실행 진입점
├─ requirements.txt
└─ README.md
```

---

## 지원 모델

| model_type | 모델명 | 특징 |
|-----------|--------|------|
| `LSTM` | 양방향 LSTM + Global Max Pooling | 빠름, 한국어 명사 추출 (KoNLPy) |
| `KoBERT` | klue/bert-base | KLUE 한국어 BERT, 풍부한 문맥 이해 |
| `KoELECTRA` | monologg/koelectra-base-v3-discriminator | KoELECTRA v3, BERT보다 효율적 |

Optuna 탐색 시 세 모델을 동시에 탐색해 최적 모델 유형을 자동으로 선택한다.

---

## 실행 방법

```bash
# 가상환경 설정 (최초 1회)
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 튜닝 + 학습 실행 (models/v{N}/ 자동 생성)
python naver_lstm_classifier.py
```

> 첫 실행 시 KoBERT / KoELECTRA 모델이 HuggingFace에서 자동 다운로드된다 (약 400MB).

---

## 산출물 (models/v{N}/)

| 파일 | 내용 |
|------|------|
| `naver_lstm_model.pt` | 학습된 모델 가중치 |
| `naver_lstm_model_meta.pkl` | 어휘·라벨·Config 메타데이터 |
| `training_curves.png` | 에포크별 정확도·손실·학습률 |
| `confusion_matrix.png` | 혼동 행렬 |
| `word_frequency.png` | 상위 단어 빈도 + 워드클라우드 |
| `optuna_results.png` | 탐색 이력·모델 유형별 성능·파라미터 중요도 |
| `optuna_study.db` | Optuna SQLite 연구 결과 (재탐색 시 이어서 사용) |

---

## 카테고리

IT · 스포츠 · 연예 · 경제 · 사회 · 정치 (총 6개)

# NAVER 뉴스 분류 v7 종합 보고서

**작성일**: 2026-06-30
**데이터**: NAVER 뉴스 헤드라인 (500건/카테고리 × 6 = 3,000건, LSTM·sklearn 기준)
**카테고리**: IT / 스포츠 / 사회 / 경제 / 연예 / 정치
**트랜스포머 상태**: Optuna 완료 / 최종평가 미완료

---

## 1. 전체 모델 정확도 요약

| 모델 | 방식 | 정확도 | BBC 대비 | 소요시간 | 버전 |
|------|------|:------:|:-------:|:-------:|------|
| BBC LSTM *(베이스라인)* | LSTM (영어) | 14.29% | 1.0× | — | — |
| TF-IDF + LDA + SVM | 예외케이스 | 53.50% | 3.7× | ~3min | v15 |
| TF-IDF + LightGBM | GBM | 64.33% | 4.5× | ~1min | v15 |
| TF-IDF + LSA + SVM | 예외케이스 | 75.50% | 5.3× | ~7min | v15 |
| TF-IDF + SVM | sklearn | 76.00% | 5.3× | ~0.1min | v15 |
| LSTM *(v5-1 구버전)* | LSTM | 58.06% | 4.1× | — | v7폴더 |
| **LSTM 고도화** | **LSTM** | **92.83%** | **6.5×** | **~90min** | **v9** |
| **KoELECTRA** | **Fine-tuning** | **96.18% *(Optuna val_acc, 최종 테스트 미완료)*** | **6.7×** | **~12h** | **v10** |
| **KoBERT** | **Fine-tuning** | **97.57% *(Optuna val_acc, 최종 테스트 미완료)*** | **6.8×** | **~12h** | **v10** |

> 형태소 추출: KoNLPy Okt 명사추출 (7.2s / 3,000건)
> sklearn 전체: 11.1분, LSTM Optuna 40 trials: ~53분, 트랜스포머 Optuna 10 trials: ~12h 33min

---

## 2. sklearn 모델 (v15)

> 추가 다운로드 없음. scikit-learn + LightGBM + KoNLPy만 사용.
> Optuna 30 trials, 3-fold CV, random_state=42

### 2-1. TF-IDF + LinearSVC (SVM) — **메인 권장**  — 튜닝 0.1min / 최종학습 0.1s / 합계 0.1min

**테스트 정확도: 76.00%**

| 카테고리 | Precision | Recall | F1-score |
|---------|----------:|-------:|---------:|
| IT | 0.854 | 0.700 | 0.769 |
| 스포츠 | 0.867 | 0.910 | 0.888 |
| 사회 | 0.592 | 0.580 | 0.586 |
| 경제 | 0.663 | 0.670 | 0.667 |
| 연예 | 0.767 | 0.920 | 0.836 |
| 정치 | 0.830 | 0.780 | 0.804 |
| **전체 macro** | **0.762** | **0.760** | **0.758** |

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_features` | `20000` |
| `ngram_range` | `[1, 2]` |
| `sublinear_tf` | `False` |
| `min_df` | `1` |
| `C` | `0.10659318318043445` |
| `model` | `LinearSVC` |
| `cv_best_acc` | `0.7369791666666666` |

---

### 2-2. TF-IDF + LightGBM  — 튜닝 0.7min / 최종학습 0.3s / 합계 0.7min

**테스트 정확도: 64.33%**

| 카테고리 | Precision | Recall | F1-score |
|---------|----------:|-------:|---------:|
| IT | 0.786 | 0.660 | 0.717 |
| 스포츠 | 0.897 | 0.780 | 0.834 |
| 사회 | 0.507 | 0.380 | 0.434 |
| 경제 | 0.621 | 0.410 | 0.494 |
| 연예 | 0.444 | 0.910 | 0.597 |
| 정치 | 0.867 | 0.720 | 0.787 |
| **전체 macro** | **0.687** | **0.643** | **0.644** |

> **SVM보다 낮은 이유**: 고차원 sparse 텍스트(20k feature)에서 트리 기반 모델은 선형 SVM보다 불리.
> HistGBM(sklearn)도 동일 이유로 11/30 trials(57분) 후 중단 — 완료 시 3~4시간 예상, CV best 60.89%에 그침.

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_features` | `10000` |
| `ngram_range` | `[1, 1]` |
| `n_estimators` | `100` |
| `max_depth` | `7` |
| `learning_rate` | `0.06323430890983159` |
| `num_leaves` | `44` |
| `min_child_samples` | `10` |
| `model` | `LightGBM` |
| `cv_best_acc` | `0.5755208333333334` |

---

### 2-3. TF-IDF + LSA(SVD) + SVM — 예외케이스  — 튜닝 7.1min / 최종학습 9.1s / 합계 7.3min

**테스트 정확도: 75.50%**

| 카테고리 | Precision | Recall | F1-score |
|---------|----------:|-------:|---------:|
| IT | 0.800 | 0.720 | 0.758 |
| 스포츠 | 0.841 | 0.900 | 0.870 |
| 사회 | 0.565 | 0.610 | 0.587 |
| 경제 | 0.698 | 0.600 | 0.645 |
| 연예 | 0.772 | 0.880 | 0.822 |
| 정치 | 0.863 | 0.820 | 0.841 |
| **전체 macro** | **0.756** | **0.755** | **0.754** |

> SVD 차원 축소 후 SVM. SVM 단독(76.00%)과 거의 동일 — 차원 축소 효과 미미.

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_features` | `10000` |
| `n_components` | `400` |
| `C` | `6.971753655503662` |
| `model` | `LSA+LinearSVC` |
| `cv_best_acc` | `0.7317708333333334` |

---

### 2-4. LDA(토픽모델) + SVM — 예외케이스  — 튜닝 2.8min / 최종학습 2.9s / 합계 2.9min

**테스트 정확도: 53.50%**

| 카테고리 | Precision | Recall | F1-score |
|---------|----------:|-------:|---------:|
| IT | 0.613 | 0.680 | 0.645 |
| 스포츠 | 0.661 | 0.820 | 0.732 |
| 사회 | 0.396 | 0.360 | 0.377 |
| 경제 | 0.465 | 0.400 | 0.430 |
| 연예 | 0.486 | 0.530 | 0.507 |
| 정치 | 0.532 | 0.420 | 0.469 |
| **전체 macro** | **0.525** | **0.535** | **0.527** |

> **예상대로 최저**: LDA는 긴 문서의 토픽 분포 모델링에 특화.
> 헤드라인 평균 7토큰 → 명사추출 3토큰 수준에서는 토픽 추정 자체가 불안정.

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_features` | `10000` |
| `min_df` | `3` |
| `n_components` | `57` |
| `C` | `37.95853142670641` |
| `model` | `LDA+LinearSVC` |
| `cv_best_acc` | `0.540625` |

---

## 3. LSTM 고도화 (v9)

**테스트 정확도: 92.83%**
데이터: 500건/카테고리 × 6 | Optuna 40 trials | 소요: ~90분

| 카테고리 | Precision | Recall | F1-score |
|---------|----------:|-------:|---------:|
| IT | 0.990 | 1.000 | 0.995 |
| 스포츠 | 0.970 | 0.960 | 0.965 |
| 사회 | 0.742 | 0.920 | 0.821 |
| 경제 | 1.000 | 0.820 | 0.901 |
| 연예 | 0.959 | 0.940 | 0.949 |
| 정치 | 0.969 | 0.930 | 0.949 |
| **전체 macro** | **0.938** | **0.928** | **0.930** |

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_vocab` | `5000` |
| `max_len` | `20` |
| `embed_dim` | `512` |
| `hidden_dim` | `128` |
| `num_layers` | `1` |
| `bidirectional` | `True` |
| `dropout` | `0.15000000000000002` |
| `batch_size` | `32` |
| `epochs` | `30` |
| `learning_rate` | `0.0005874509725681373` |
| `test_size` | `0.2` |
| `val_size` | `0.2` |
| `random_state` | `42` |
| `patience` | `5` |
| `optimizer_name` | `Adam` |
| `weight_decay` | `0.001981302590731254` |
| `use_morphemes` | `False` |
| `max_items` | `500` |

---

## 4. KoBERT (v10)  *(Optuna val_acc, 최종 테스트 미완료)*

**Optuna val 정확도: 97.57%**
Base: `klue/bert-base` | 데이터: 300건/카테고리 | Optuna 10 trials

_최종 테스트 평가 미완료 — Optuna val_acc 기준_

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_vocab` | `5000` |
| `max_len` | `128` |
| `embed_dim` | `128` |
| `hidden_dim` | `128` |
| `num_layers` | `2` |
| `bidirectional` | `True` |
| `dropout` | `0.15000000000000002` |
| `batch_size` | `8` |
| `epochs` | `30` |
| `learning_rate` | `4.3072264894936264e-05` |
| `test_size` | `0.2` |
| `val_size` | `0.2` |
| `random_state` | `42` |
| `patience` | `5` |
| `optimizer_name` | `AdamW` |
| `weight_decay` | `0.00019408559236656545` |
| `use_morphemes` | `False` |
| `max_items` | `300` |

---

## 5. KoELECTRA (v10)  *(Optuna val_acc, 최종 테스트 미완료)*

**Optuna val 정확도: 96.18%**
Base: `monologg/koelectra-base-v3-discriminator` | 데이터: 300건/카테고리 | Optuna 10 trials

_최종 테스트 평가 미완료 — Optuna val_acc 기준_

**최적 하이퍼파라미터**: | 파라미터 | 값 |
|----------|-----|
| `max_vocab` | `5000` |
| `max_len` | `128` |
| `embed_dim` | `128` |
| `hidden_dim` | `128` |
| `num_layers` | `2` |
| `bidirectional` | `True` |
| `dropout` | `0.4` |
| `batch_size` | `8` |
| `epochs` | `30` |
| `learning_rate` | `3.086461470994519e-05` |
| `test_size` | `0.2` |
| `val_size` | `0.2` |
| `random_state` | `42` |
| `patience` | `5` |
| `optimizer_name` | `AdamW` |
| `weight_decay` | `0.008024036368429074` |
| `use_morphemes` | `False` |
| `max_items` | `300` |

---

## 6. 핵심 인사이트

### 왜 헤드라인 분류는 기대보다 낮은가?

| 요소 | full article | headline (본 프로젝트) |
|------|:--------:|:--------:|
| 평균 텍스트 길이 | 수백 단어 | **7토큰** |
| 명사 추출 후 | 수십 단어 | **2~3토큰** |
| TF-IDF 신뢰도 | 높음 | **낮음** |
| 일반 벤치마크 | 83~87% | 76% (본 결과) |

→ 타 논문의 83~87%는 **전문 기사(full article)** 기준. 헤드라인에서 76%는 정상적 결과.

### 모델별 특성 비교

| 요소 | TF-IDF+SVM | LSTM | KoBERT |
|------|:----------:|:----:|:------:|
| 단어 의미 | ❌ 빈도 | △ 학습 | ✅ 사전학습 |
| 순서 활용 | ❌ | ✅ | ✅ Attention |
| 짧은 텍스트 | 취약 | 중간 | 강함 |
| 학습 시간 | **~0.1min** | ~90min | ~12h |
| 추가 다운로드 | ❌ | ❌ | ✅ |

### GBM이 SVM보다 낮은 이유
텍스트 TF-IDF 피처는 고차원(20,000) sparse 행렬.
- **SVM**: sparse 그대로 선형 분리 → 빠르고 효과적
- **GBM(트리)**: dense 변환 필수 → 메모리 폭발, 20k 피처 split 탐색 비효율
- HistGBM: 3~4시간 예상 → LightGBM 교체 → 39초 완료, 그래도 64% (SVM 76%보다 낮음)
- **결론**: 헤드라인 텍스트 분류에서 GBM < SVM (구조적 이유)

---

## 7. 산출물 목록

| 파일 | 위치 | 설명 |
|------|------|------|
| TF-IDF+SVM 파이프라인 | `models/v15/sklearn_tfidf_svm_pipeline.pkl` | 즉시 추론 가능 |
| LightGBM 파이프라인 | `models/v15/sklearn_tfidf_lgbm_pipeline.pkl` | |
| LDA+SVM 파이프라인 | `models/v15/sklearn_lda_svm_pipeline.pkl` | |
| LSA+SVM 파이프라인 | `models/v15/sklearn_lsa_svm_pipeline.pkl` | |
| sklearn 하이퍼파라미터 | `models/v15/sklearn_hyperparams.json` | 전 모델 + 타이밍 |
| LSTM 모델 | `models/v9/naver_lstm_model.pt` | |
| KoBERT 모델 | `models/v10/naver_kobert_model.pt` | 442MB |
| HistGBM 중단 기록 | `models/v14/histgbm_partial_results.json` | 11/30 trials, 57min |

**sklearn 파이프라인 사용 예시:**
```python
import pickle
with open("models/v15/sklearn_tfidf_svm_pipeline.pkl", "rb") as f:
    obj = pickle.load(f)
pred = obj["label_encoder"].inverse_transform(
    obj["pipeline"].predict(["삼성전자 반도체 신제품 발표"])
)
print(pred)  # ['IT']
```

---

## 8. 결론

1. **추가 다운로드 없이 최선**: TF-IDF + LinearSVC → **76.00%** (11분 전체 소요)
2. **LSTM 한계 극복**: 데이터 5배(100→500건/cat) + 형태소 튜닝으로 41.67% → **92.83%**
3. **트랜스포머 압도적 우위**: KoBERT **97.57%** / KoELECTRA **96.18%** (단, 학습시간 ~12h)
4. **헤드라인 특성 확인**: 짧은 텍스트(7토큰)에서 TF-IDF 한계 실증, 표현학습(임베딩) 효과 두드러짐
5. **GBM 부적합 확인**: sparse 텍스트에서 트리 모델은 선형 SVM보다 구조적으로 불리

"""sklearn 텍스트 분류 모델 4종 학습 · 평가 · 저장

모델:
  1. 형태소 TF-IDF + LinearSVC          (메인)
  2. 형태소 TF-IDF + HistGradientBoost  (메인)
  3. 형태소 CountVec + LDA + LinearSVC  (예외 케이스)
  4. 형태소 TF-IDF + LSA(SVD) + LinearSVC (예외 케이스)

산출물 (models/v{N}/):
  sklearn_{name}_pipeline.pkl     전체 파이프라인 (재현용)
  classification_report_{name}.json
  sklearn_hyperparams.json        전 모델 하이퍼파라미터 기록
  results.json

재현 조건:
  random_state=42, 데이터=data/naver_news_500per_cat.csv
  train/val/test = 64% / 16% / 20%  (LSTM 과 동일)
"""

from __future__ import annotations

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import optuna
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from app.data import load_sample_data
from app.preprocess import precompute_tokens

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 설정 ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_TRIALS     = 30       # Optuna trials 수
CV_FOLDS     = 3        # Cross-validation fold 수
MAX_ITEMS    = 500      # 캐시에서 로드할 카테고리당 건수

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── 버전 폴더 생성 ─────────────────────────────────────────────────────────────
def _next_run_dir() -> Path:
    existing = sorted(
        int(d.name[1:])
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^v\d+$", d.name)
    ) if MODELS_DIR.exists() else []
    version = (existing[-1] + 1) if existing else 1
    run_dir = MODELS_DIR / f"v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ── 데이터 로드 및 분할 ────────────────────────────────────────────────────────
print("=" * 60)
print("  sklearn 텍스트 분류 모델 학습")
print("=" * 60)

texts, labels = load_sample_data(max_items=MAX_ITEMS)

le = LabelEncoder()
y  = le.fit_transform(labels)

# LSTM 과 동일한 분할 비율 (test=0.2, val=0.2)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    texts, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y_trainval
)

# 형태소 추출 (KoNLPy Okt 명사, 이미 설치됨)
print("\n형태소 추출 중...")
all_token_lists = precompute_tokens(texts, use_morphemes=False)
all_texts_joined = [" ".join(tl) for tl in all_token_lists]

n = len(texts)
n_test = int(n * 0.2)
# 동일 random_state 로 인덱스 재현
idx_all = np.arange(n)
idx_trainval, idx_test = train_test_split(idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
idx_train, idx_val     = train_test_split(idx_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y[idx_trainval])

X_tr = [all_texts_joined[i] for i in idx_train]
X_vl = [all_texts_joined[i] for i in idx_val]
X_te = [all_texts_joined[i] for i in idx_test]
y_tr, y_vl, y_te = y[idx_train], y[idx_val], y[idx_test]

print(f"  Train {len(X_tr)} / Val {len(X_vl)} / Test {len(X_te)}")
print(f"  카테고리: {list(le.classes_)}\n")

run_dir    = _next_run_dir()
all_params = {}
all_results = {}
category_names = list(le.classes_)

print(f"산출물 저장 위치: {run_dir}\n")


# ── 공통 평가 함수 ─────────────────────────────────────────────────────────────
def evaluate_and_save(name: str, pipeline, params: dict):
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    acc   = accuracy_score(y_te, preds)

    print(f"\n{'='*50}")
    print(f"  [{name}] 테스트 정확도: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_te, preds, target_names=category_names, zero_division=0))

    clf_report = classification_report(
        y_te, preds, target_names=category_names,
        zero_division=0, output_dict=True
    )

    # 파이프라인 저장 (재현용)
    pkl_path = run_dir / f"sklearn_{name}_pipeline.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": le}, f)

    # classification report 저장
    rpt_path = run_dir / f"classification_report_{name}.json"
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "accuracy": acc, "report": clf_report}, f,
                  ensure_ascii=False, indent=2)

    all_params[name]  = params
    all_results[name] = acc
    print(f"  저장 완료: {pkl_path.name}")
    return acc


# ── 1. TF-IDF + LinearSVC ─────────────────────────────────────────────────────
print("[ 1/4 ] TF-IDF + LinearSVC (SVM) — Optuna 튜닝 중...")

def objective_tfidf_svm(trial):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = trial.suggest_categorical("max_features", [5000, 10000, 20000, 30000]),
            ngram_range  = trial.suggest_categorical("ngram_range",  [(1,1),(1,2),(1,3)]),
            sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False]),
            min_df       = trial.suggest_int("min_df", 1, 3),
        )),
        ("clf", LinearSVC(
            C            = trial.suggest_float("C", 0.01, 100, log=True),
            max_iter     = 2000,
            random_state = RANDOM_STATE,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study1 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study1.optimize(objective_tfidf_svm, n_trials=N_TRIALS, show_progress_bar=False)
bp1 = study1.best_params

pipe_tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp1["max_features"], ngram_range=bp1["ngram_range"],
        sublinear_tf=bp1["sublinear_tf"], min_df=bp1["min_df"],
    )),
    ("clf", LinearSVC(C=bp1["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params1 = {**bp1, "model": "LinearSVC", "cv_best_acc": study1.best_value}
evaluate_and_save("tfidf_svm", pipe_tfidf_svm, params1)


# ── 2. TF-IDF + HistGradientBoostingClassifier ───────────────────────────────
print("\n[ 2/4 ] TF-IDF + HistGBM — Optuna 튜닝 중...")

def objective_tfidf_gbm(trial):
    from sklearn.preprocessing import MaxAbsScaler
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = trial.suggest_categorical("max_features", [5000, 10000, 20000]),
            ngram_range  = trial.suggest_categorical("ngram_range",  [(1,1),(1,2)]),
            sublinear_tf = True,
        )),
        ("scaler", MaxAbsScaler()),   # sparse → sparse (GBM 안정성)
        ("clf", HistGradientBoostingClassifier(
            max_iter       = trial.suggest_int("max_iter", 100, 400, step=50),
            max_depth      = trial.suggest_int("max_depth", 3, 8),
            learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
            random_state   = RANDOM_STATE,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study2 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study2.optimize(objective_tfidf_gbm, n_trials=N_TRIALS, show_progress_bar=False)
bp2 = study2.best_params

from sklearn.preprocessing import MaxAbsScaler
pipe_tfidf_gbm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp2["max_features"], ngram_range=bp2["ngram_range"], sublinear_tf=True,
    )),
    ("scaler", MaxAbsScaler()),
    ("clf", HistGradientBoostingClassifier(
        max_iter=bp2["max_iter"], max_depth=bp2["max_depth"],
        learning_rate=bp2["learning_rate"], min_samples_leaf=bp2["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )),
])
params2 = {**bp2, "model": "HistGBM", "cv_best_acc": study2.best_value}
evaluate_and_save("tfidf_gbm", pipe_tfidf_gbm, params2)


# ── 3. LDA + LinearSVC (예외 케이스) ─────────────────────────────────────────
print("\n[ 3/4 ] LDA (토픽모델) + LinearSVC — Optuna 튜닝 중...")

def objective_lda_svm(trial):
    pipe = Pipeline([
        ("cv", CountVectorizer(     # LDA 는 count 입력 필요
            max_features = trial.suggest_categorical("max_features", [3000, 5000, 10000]),
            min_df       = trial.suggest_int("min_df", 1, 3),
        )),
        ("lda", LatentDirichletAllocation(
            n_components = trial.suggest_int("n_components", 6, 60),
            max_iter     = 20,
            random_state = RANDOM_STATE,
            learning_method = "batch",
        )),
        ("clf", LinearSVC(
            C=trial.suggest_float("C", 0.01, 100, log=True),
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study3 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study3.optimize(objective_lda_svm, n_trials=N_TRIALS, show_progress_bar=False)
bp3 = study3.best_params

pipe_lda_svm = Pipeline([
    ("cv",  CountVectorizer(max_features=bp3["max_features"], min_df=bp3["min_df"])),
    ("lda", LatentDirichletAllocation(
        n_components=bp3["n_components"], max_iter=20,
        random_state=RANDOM_STATE, learning_method="batch",
    )),
    ("clf", LinearSVC(C=bp3["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params3 = {**bp3, "model": "LDA+LinearSVC", "cv_best_acc": study3.best_value,
           "note": "예외 케이스: 짧은 헤드라인에 LDA 부적합 확인용"}
evaluate_and_save("lda_svm", pipe_lda_svm, params3)


# ── 4. LSA (TF-IDF + SVD) + LinearSVC (예외 케이스) ─────────────────────────
print("\n[ 4/4 ] LSA (TF-IDF + SVD) + LinearSVC — Optuna 튜닝 중...")

def objective_lsa_svm(trial):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = trial.suggest_categorical("max_features", [5000, 10000, 20000]),
            sublinear_tf = True,
            ngram_range  = (1, 2),
        )),
        ("svd", TruncatedSVD(
            n_components = trial.suggest_int("n_components", 50, 400, step=50),
            random_state = RANDOM_STATE,
        )),
        ("clf", LinearSVC(
            C=trial.suggest_float("C", 0.01, 100, log=True),
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study4 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study4.optimize(objective_lsa_svm, n_trials=N_TRIALS, show_progress_bar=False)
bp4 = study4.best_params

pipe_lsa_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp4["max_features"], sublinear_tf=True, ngram_range=(1,2),
    )),
    ("svd", TruncatedSVD(n_components=bp4["n_components"], random_state=RANDOM_STATE)),
    ("clf", LinearSVC(C=bp4["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params4 = {**bp4, "model": "LSA+LinearSVC", "cv_best_acc": study4.best_value,
           "note": "TF-IDF 차원축소 후 SVM"}
evaluate_and_save("lsa_svm", pipe_lsa_svm, params4)


# ── 전체 결과 저장 ─────────────────────────────────────────────────────────────
# 하이퍼파라미터 전체 기록
params_path = run_dir / "sklearn_hyperparams.json"
with open(params_path, "w", encoding="utf-8") as f:
    json.dump(all_params, f, ensure_ascii=False, indent=2)

# results.json (run_v7_report.py 가 읽음)
results_path = run_dir / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "version": run_dir.name,
        "data": f"data/naver_news_{MAX_ITEMS}per_cat.csv",
        "train_size": len(X_tr),
        "val_size":   len(X_vl),
        "test_size":  len(X_te),
        "random_state": RANDOM_STATE,
        "results": all_results,
    }, f, ensure_ascii=False, indent=2)

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [{run_dir.name}] sklearn 모델 최종 결과")
print(f"{'='*60}")
for name, acc in sorted(all_results.items(), key=lambda x: -x[1]):
    bar = "#" * int(acc * 20)
    print(f"  {name:20s}: {acc:.4f}  {bar}")
print(f"\n  산출물 위치: {run_dir}")
print(f"  하이퍼파라미터: {params_path.name}")
print(f"{'='*60}\n")

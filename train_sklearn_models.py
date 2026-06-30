"""sklearn 텍스트 분류 모델 4종 학습 · 평가 · 저장

모델:
  1. 형태소 TF-IDF + LinearSVC          (메인)
  2. 형태소 TF-IDF + LightGBM           (메인, sparse 네이티브 지원)
  3. 형태소 CountVec + LDA + LinearSVC  (예외 케이스)
  4. 형태소 TF-IDF + LSA(SVD) + LinearSVC (예외 케이스)

산출물 (models/v{N}/):
  sklearn_{name}_pipeline.pkl     전체 파이프라인 (재현용)
  classification_report_{name}.json  (타이밍 포함)
  sklearn_hyperparams.json        전 모델 하이퍼파라미터 + 타이밍 기록
  results.json

재현 조건:
  random_state=42, 데이터=data/naver_news_500per_cat.csv
  train/val/test = 64% / 16% / 20%  (LSTM 과 동일)
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from app.data import load_sample_data
from app.preprocess import precompute_tokens

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 설정 ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_TRIALS     = 30
CV_FOLDS     = 3
MAX_ITEMS    = 500

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SCRIPT_START = time.time()


def _elapsed(since: float) -> str:
    s = time.time() - since
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def _make_trial_callback(label: str, n_trials: int, t_start: float):
    from tqdm import tqdm
    pbar = tqdm(total=n_trials, desc=f"  [{label}]", ncols=72,
                file=sys.stdout, dynamic_ncols=False)

    def callback(study, trial):
        pbar.update(1)
        elapsed = time.time() - t_start
        pbar.set_postfix({
            "best": f"{study.best_value:.4f}",
            "cur":  f"{trial.value:.4f}" if trial.value is not None else "pruned",
            "t":    f"{elapsed:.0f}s",
        })
        sys.stdout.flush()

    callback._pbar = pbar
    return callback


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


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  sklearn 텍스트 분류 모델 학습")
print("=" * 60)

t0 = time.time()
texts, labels = load_sample_data(max_items=MAX_ITEMS)
le = LabelEncoder()
y  = le.fit_transform(labels)

idx_all = np.arange(len(texts))
idx_trainval, idx_test = train_test_split(
    idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y[idx_trainval])

print("\n형태소 추출 중...")
t_tok = time.time()
all_token_lists  = precompute_tokens(texts, use_morphemes=False)
all_texts_joined = [" ".join(tl) for tl in all_token_lists]
tok_time = time.time() - t_tok

X_tr = [all_texts_joined[i] for i in idx_train]
X_vl = [all_texts_joined[i] for i in idx_val]
X_te = [all_texts_joined[i] for i in idx_test]
y_tr, y_vl, y_te = y[idx_train], y[idx_val], y[idx_test]

print(f"  Train {len(X_tr)} / Val {len(X_vl)} / Test {len(X_te)}  (형태소 추출: {tok_time:.1f}s)")
print(f"  카테고리: {list(le.classes_)}\n")

run_dir      = _next_run_dir()
all_params   = {}
all_results  = {}
all_timing   = {}
category_names = list(le.classes_)

print(f"산출물 저장 위치: {run_dir}\n")


# ── 공통 평가 함수 ─────────────────────────────────────────────────────────────
def evaluate_and_save(name: str, pipeline, params: dict,
                      tune_sec: float, note: str = ""):
    t_train = time.time()
    pipeline.fit(X_tr, y_tr)
    train_sec = time.time() - t_train

    t_pred = time.time()
    preds  = pipeline.predict(X_te)
    pred_sec = time.time() - t_pred

    acc = accuracy_score(y_te, preds)
    total_sec = tune_sec + train_sec

    print(f"\n{'='*50}")
    print(f"  [{name}] 테스트 정확도: {acc:.4f}  "
          f"(튜닝 {tune_sec/60:.1f}min + 최종학습 {train_sec:.1f}s)")
    print(f"{'='*50}")
    print(classification_report(y_te, preds, target_names=category_names, zero_division=0))

    clf_report = classification_report(
        y_te, preds, target_names=category_names,
        zero_division=0, output_dict=True
    )

    timing = {
        "tokenization_sec": round(tok_time, 2),
        "tuning_sec":       round(tune_sec, 2),
        "tuning_min":       round(tune_sec / 60, 2),
        "final_train_sec":  round(train_sec, 2),
        "inference_sec":    round(pred_sec, 4),
        "total_sec":        round(total_sec, 2),
        "total_min":        round(total_sec / 60, 2),
    }

    pkl_path = run_dir / f"sklearn_{name}_pipeline.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": le}, f)

    rpt_path = run_dir / f"classification_report_{name}.json"
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": name, "accuracy": acc,
            "report": clf_report, "timing": timing,
            "note": note,
        }, f, ensure_ascii=False, indent=2)

    all_params[name]  = {**params, "timing": timing}
    all_results[name] = acc
    all_timing[name]  = timing
    print(f"  저장 완료: {pkl_path.name}")
    return acc


# ── 1. TF-IDF + LinearSVC ─────────────────────────────────────────────────────
print("[ 1/4 ] TF-IDF + LinearSVC (SVM) --Optuna 튜닝 중...")
t1 = time.time()

def objective_tfidf_svm(trial):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = trial.suggest_categorical("max_features", [5000, 10000, 20000, 30000]),
            ngram_range  = trial.suggest_categorical("ngram_range",  [(1,1),(1,2),(1,3)]),
            sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False]),
            min_df       = trial.suggest_int("min_df", 1, 3),
        )),
        ("clf", LinearSVC(
            C=trial.suggest_float("C", 0.01, 100, log=True),
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study1 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
cb1 = _make_trial_callback("tfidf_svm", N_TRIALS, t1)
study1.optimize(objective_tfidf_svm, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[cb1])
cb1._pbar.close()
tune1 = time.time() - t1
bp1 = study1.best_params

pipe_tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp1["max_features"], ngram_range=bp1["ngram_range"],
        sublinear_tf=bp1["sublinear_tf"], min_df=bp1["min_df"],
    )),
    ("clf", LinearSVC(C=bp1["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params1 = {**bp1, "model": "LinearSVC", "cv_best_acc": study1.best_value}
evaluate_and_save("tfidf_svm", pipe_tfidf_svm, params1, tune1,
                  note="추가 다운로드 없음. 텍스트 분류 정석 베이스라인.")


# ── 2. TF-IDF + LightGBM ─────────────────────────────────────────────────────
print("\n[ 2/4 ] TF-IDF + LightGBM --Optuna 튜닝 중...")
print("  (LightGBM: sparse 네이티브 지원, HistGBM 대비 ~10x 빠름)")
t2 = time.time()

def objective_tfidf_lgbm(trial):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = trial.suggest_categorical("max_features", [5000, 10000, 20000, 30000]),
            ngram_range  = trial.suggest_categorical("ngram_range",  [(1,1),(1,2)]),
            sublinear_tf = True,
        )),
        ("clf", LGBMClassifier(
            n_estimators    = trial.suggest_int("n_estimators", 100, 500, step=100),
            max_depth       = trial.suggest_int("max_depth", 3, 8),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 15, 127),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50),
            random_state    = RANDOM_STATE,
            verbose         = -1,
            n_jobs          = -1,
        )),
    ])
    return cross_val_score(pipe, X_tr, y_tr, cv=CV_FOLDS, scoring="accuracy").mean()

study2 = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
cb2 = _make_trial_callback("tfidf_lgbm", N_TRIALS, t2)
study2.optimize(objective_tfidf_lgbm, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[cb2])
cb2._pbar.close()
tune2 = time.time() - t2
bp2 = study2.best_params

pipe_tfidf_lgbm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp2["max_features"], ngram_range=bp2["ngram_range"], sublinear_tf=True,
    )),
    ("clf", LGBMClassifier(
        n_estimators=bp2["n_estimators"], max_depth=bp2["max_depth"],
        learning_rate=bp2["learning_rate"], num_leaves=bp2["num_leaves"],
        min_child_samples=bp2["min_child_samples"],
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )),
])
params2 = {**bp2, "model": "LightGBM", "cv_best_acc": study2.best_value}
evaluate_and_save("tfidf_lgbm", pipe_tfidf_lgbm, params2, tune2,
                  note="LightGBM 4.6.0. sparse TF-IDF 네이티브 지원. HistGBM 대비 10x 빠름.")


# ── 3. LDA + LinearSVC (예외 케이스) ─────────────────────────────────────────
print("\n[ 3/4 ] LDA (토픽모델) + LinearSVC --Optuna 튜닝 중...")
print("  (예외 케이스: 짧은 헤드라인에 LDA 부적합 여부 실증)")
t3 = time.time()

def objective_lda_svm(trial):
    pipe = Pipeline([
        ("cv", CountVectorizer(
            max_features = trial.suggest_categorical("max_features", [3000, 5000, 10000]),
            min_df       = trial.suggest_int("min_df", 1, 3),
        )),
        ("lda", LatentDirichletAllocation(
            n_components    = trial.suggest_int("n_components", 6, 60),
            max_iter        = 20,
            random_state    = RANDOM_STATE,
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
cb3 = _make_trial_callback("lda_svm", N_TRIALS, t3)
study3.optimize(objective_lda_svm, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[cb3])
cb3._pbar.close()
tune3 = time.time() - t3
bp3 = study3.best_params

pipe_lda_svm = Pipeline([
    ("cv",  CountVectorizer(max_features=bp3["max_features"], min_df=bp3["min_df"])),
    ("lda", LatentDirichletAllocation(
        n_components=bp3["n_components"], max_iter=20,
        random_state=RANDOM_STATE, learning_method="batch",
    )),
    ("clf", LinearSVC(C=bp3["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params3 = {**bp3, "model": "LDA+LinearSVC", "cv_best_acc": study3.best_value}
evaluate_and_save("lda_svm", pipe_lda_svm, params3, tune3,
                  note="예외 케이스: LDA는 긴 문서 전용, 헤드라인(평균 7토큰)에 부적합 확인용")


# ── 4. LSA (TF-IDF + SVD) + LinearSVC (예외 케이스) ─────────────────────────
print("\n[ 4/4 ] LSA (TF-IDF + SVD) + LinearSVC --Optuna 튜닝 중...")
print("  (예외 케이스: 차원 축소 후 SVM)")
t4 = time.time()

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
cb4 = _make_trial_callback("lsa_svm", N_TRIALS, t4)
study4.optimize(objective_lsa_svm, n_trials=N_TRIALS, show_progress_bar=False, callbacks=[cb4])
cb4._pbar.close()
tune4 = time.time() - t4
bp4 = study4.best_params

pipe_lsa_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=bp4["max_features"], sublinear_tf=True, ngram_range=(1,2),
    )),
    ("svd", TruncatedSVD(n_components=bp4["n_components"], random_state=RANDOM_STATE)),
    ("clf", LinearSVC(C=bp4["C"], max_iter=2000, random_state=RANDOM_STATE)),
])
params4 = {**bp4, "model": "LSA+LinearSVC", "cv_best_acc": study4.best_value}
evaluate_and_save("lsa_svm", pipe_lsa_svm, params4, tune4,
                  note="TF-IDF 희소 벡터를 SVD로 차원 축소 후 SVM. 정보 손실 발생.")


# ── 전체 결과 저장 ─────────────────────────────────────────────────────────────
total_sec = time.time() - SCRIPT_START

params_path = run_dir / "sklearn_hyperparams.json"
with open(params_path, "w", encoding="utf-8") as f:
    json.dump(all_params, f, ensure_ascii=False, indent=2)

results_path = run_dir / "results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({
        "version":      run_dir.name,
        "data":         f"data/naver_news_{MAX_ITEMS}per_cat.csv",
        "train_size":   len(X_tr),
        "val_size":     len(X_vl),
        "test_size":    len(X_te),
        "random_state": RANDOM_STATE,
        "n_trials":     N_TRIALS,
        "results":      all_results,
        "timing":       {
            **all_timing,
            "tokenization_sec": round(tok_time, 2),
            "total_script_sec": round(total_sec, 2),
            "total_script_min": round(total_sec / 60, 2),
        },
        "histgbm_reference": "models/v14/histgbm_partial_results.json",
    }, f, ensure_ascii=False, indent=2)

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  [{run_dir.name}] sklearn 모델 최종 결과")
print(f"{'='*60}")
for name, acc in sorted(all_results.items(), key=lambda x: -x[1]):
    t = all_timing[name]
    bar = "#" * int(acc * 20)
    print(f"  {name:16s}: {acc:.4f}  {bar}  ({t['total_min']:.1f}min)")
print(f"\n  형태소 추출: {tok_time:.1f}s")
print(f"  전체 소요:   {total_sec/60:.1f}min")
print(f"  산출물 위치: {run_dir}")
print(f"{'='*60}\n")

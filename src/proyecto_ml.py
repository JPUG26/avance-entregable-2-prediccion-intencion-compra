"""Utilidades reproducibles para la seccion 1 del Entregable II.

Prediccion de la intencion de compra (Online Shoppers Purchasing Intention,
UCI #468). Todo el preprocesamiento se ajusta dentro de los folds de
entrenamiento mediante Pipeline/ColumnTransformer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.svm import SVC

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "clipboard_1790819347953_online_shoppers_intention.csv"

NUMERICAS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]
LOG_VARS = NUMERICAS
NOMINALES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
BINARIAS = ["Weekend"]
ESPECIALES = ["SpecialDay"]
PREDICTORAS = NUMERICAS + NOMINALES + BINARIAS + ESPECIALES
OBJETIVO = "Revenue"

DIR_RES = RAIZ / "resultados"
DIR_FIG = DIR_RES / "figuras"


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str | Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df = df.copy()
    for c in ["Weekend", "Revenue"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.upper().map(
                {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
            )
    return df


def auditar(df: pd.DataFrame, ruta_csv: str | Path = CSV) -> dict:
    ruta_csv = Path(ruta_csv)
    h = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    aud = {
        "archivo": ruta_csv.name,
        "sha256": h,
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados_exactos": int(df.duplicated().sum()),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "faltantes_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "clases": {str(k): int(v) for k, v in df[OBJETIVO].value_counts().sort_index().items()},
        "proporcion_compra": float(df[OBJETIVO].mean()),
        "categorias": {
            c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
            for c in NOMINALES + BINARIAS + ESPECIALES
        },
        "rango_numericas": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "media": float(df[c].mean()),
                "desv": float(df[c].std()),
                "ceros": int((df[c] == 0).sum()),
                "asimetria": float(df[c].skew()),
            }
            for c in NUMERICAS + ESPECIALES
        },
        "valores_especialday": sorted(float(v) for v in df["SpecialDay"].unique()),
        "version_python": platform.python_version(),
        "version_sklearn": sklearn.__version__,
        "semilla": SEMILLA,
    }
    return aud


def tabla_auditoria(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in PREDICTORAS + [OBJETIVO]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            filas.append(
                {
                    "variable": c,
                    "tipo": "numerica",
                    "n_unicos": int(s.nunique()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "media": float(s.mean()),
                    "nulos": int(s.isna().sum()),
                }
            )
        else:
            vc = s.value_counts()
            filas.append(
                {
                    "variable": c,
                    "tipo": "categorica",
                    "n_unicos": int(s.nunique()),
                    "min": np.nan,
                    "max": np.nan,
                    "media": np.nan,
                    "nulos": int(s.isna().sum()),
                    "categoria_mas_frecuente": str(vc.index[0]),
                    "frecuencia_max": int(vc.iloc[0]),
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def particionar(df: pd.DataFrame):
    """Particion estratificada 60/20/20 sobre datos sin duplicados exactos."""
    df_u = df.drop_duplicates().reset_index(drop=True)
    X = df_u[PREDICTORAS].copy()
    y = df_u[OBJETIVO].astype(int).copy()
    idx = np.arange(len(df_u))
    idx_tr, idx_resto = train_test_split(
        idx, test_size=0.40, stratify=y, random_state=SEMILLA
    )
    idx_val, idx_te = train_test_split(
        idx_resto, test_size=0.50, stratify=y.iloc[idx_resto], random_state=SEMILLA
    )
    return df_u, X, y, idx_tr, idx_val, idx_te


def resumen_particiones(y: pd.Series, idx_tr, idx_val, idx_te) -> pd.DataFrame:
    filas = []
    for nombre, ix in [
        ("Entrenamiento", idx_tr),
        ("Validacion", idx_val),
        ("Test", idx_te),
    ]:
        yy = y.iloc[ix]
        filas.append(
            {
                "particion": nombre,
                "n": int(len(ix)),
                "positivos": int(yy.sum()),
                "negativos": int(len(yy) - yy.sum()),
                "proporcion_compra": float(yy.mean()),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
class TransformadorLog(BaseEstimator, TransformerMixin):
    """Aplica log1p opcionalmente; la decision se ajusta solo con train/CV."""

    def __init__(self, usar_log: bool = True):
        self.usar_log = usar_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(X) if self.usar_log else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def construir_preprocesador(
    usar_log: bool = True,
    incluir_pagevalues: bool = True,
    incluir_specialday: bool = True,
):
    numericas = list(LOG_VARS if incluir_pagevalues else [c for c in LOG_VARS if c != "PageValues"])
    num_pipe = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("log", TransformadorLog(usar_log=usar_log)),
            ("escalar", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformadores = [
        ("num", num_pipe, numericas),
        ("cat", cat_pipe, NOMINALES),
        ("bin", "passthrough", BINARIAS),
    ]
    if incluir_specialday:
        transformadores.append(
            (
                "esp",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                ESPECIALES,
            )
        )
    return ColumnTransformer(transformadores, remainder="drop")


def construir_modelos(incluir_pagevalues: bool = True, incluir_specialday: bool = True):
    """Devuelve (modelos, mallas) con transformacion log1p como hiperparametro."""
    pre = construir_preprocesador
    modelos = {}
    mallas = {}

    modelos["Regresion_logistica"] = Pipeline(
        [
            ("pre", pre(incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday)),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEMILLA)),
        ]
    )
    mallas["Regresion_logistica"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"],
    }

    modelos["kNN"] = Pipeline(
        [
            ("pre", pre(incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday)),
            ("clf", KNeighborsClassifier()),
        ]
    )
    mallas["kNN"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__n_neighbors": [5, 15, 31],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }

    modelos["RandomForest"] = Pipeline(
        [
            ("pre", pre(incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, random_state=SEMILLA, n_jobs=1
                ),
            ),
        ]
    )
    mallas["RandomForest"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__max_depth": [None, 12],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced"],
    }

    modelos["MLP"] = Pipeline(
        [
            ("pre", pre(incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday)),
            (
                "clf",
                MLPClassifier(
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    mallas["MLP"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "clf__alpha": [0.0001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01],
    }

    modelos["SVM"] = Pipeline(
        [
            ("pre", pre(incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday)),
            ("clf", SVC(probability=False, random_state=SEMILLA)),
        ]
    )
    mallas["SVM"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__kernel": ["linear", "rbf"],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01],
        "clf__class_weight": [None, "balanced"],
    }
    return modelos, mallas


def contar_combinaciones(mallas: dict) -> pd.DataFrame:
    filas = []
    for m, g in mallas.items():
        n = 1
        for v in g.values():
            n *= len(v)
        filas.append({"modelo": m, "combinaciones": n})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Puntuaciones y metricas
# ---------------------------------------------------------------------------
def puntuacion(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def predecir(modelo, X):
    """Umbral nativo: 0.5 probabilistico, 0 decision_function."""
    if hasattr(modelo, "predict_proba"):
        return (modelo.predict_proba(X)[:, 1] >= 0.5).astype(int)
    return (modelo.decision_function(X) >= 0).astype(int)


def metricas(y_true, y_pred, y_score) -> dict:
    return {
        "AP": float(average_precision_score(y_true, y_score)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def bootstrap_ic(y_true, y_pred, y_score, n_boot=2000, semilla=SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    registros = []
    for _ in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        ix = np.concatenate([ip, ineg])
        yt, yp, ys = y_true[ix], y_pred[ix], y_score[ix]
        if yt.min() == yt.max():
            continue
        registros.append(metricas(yt, yp, ys))
    b = pd.DataFrame(registros)
    return pd.DataFrame(
        {
            "metrica": b.columns,
            "media_bootstrap": b.mean().values,
            "ic_inferior": b.quantile(0.025).values,
            "ic_superior": b.quantile(0.975).values,
        }
    )


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
@dataclass
class ResultadoFamilia:
    nombre: str
    mejor_params: dict
    cv_ap: float
    cv_std: float
    modelo: object
    tiempo: float


def entrenar_familia(nombre, modelo, malla, X_tr, y_tr, cv, n_jobs=-1):
    t0 = time.time()
    gs = GridSearchCV(
        modelo,
        malla,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)
    dt = time.time() - t0
    best_i = gs.best_index_
    res = ResultadoFamilia(
        nombre=nombre,
        mejor_params=gs.best_params_,
        cv_ap=float(gs.best_score_),
        cv_std=float(gs.cv_results_["std_test_score"][best_i]),
        modelo=gs.best_estimator_,
        tiempo=dt,
    )
    cv_df = pd.DataFrame(gs.cv_results_).copy()
    cv_df["modelo"] = nombre
    cv_df["mejor"] = cv_df.index == best_i
    return res, cv_df


def evaluar(res: ResultadoFamilia, X, y, particion: str) -> dict:
    y_pred = predecir(res.modelo, X)
    y_score = puntuacion(res.modelo, X)
    d = metricas(y, y_pred, y_score)
    d.update({"modelo": res.nombre, "particion": particion})
    return d


def matriz_confusion(res: ResultadoFamilia, X, y):
    return confusion_matrix(y, predecir(res.modelo, X), labels=[0, 1])

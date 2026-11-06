"""Entrenamiento y evaluacion de los cinco modelos del Entregable II (seccion 1).

Ejecuta:
  - busqueda en rejilla con validacion cruzada estratificada de 5 folds sobre train;
  - evaluacion en entrenamiento, validacion y test (test una sola vez);
  - intervalos de confianza por bootstrap estratificado;
  - sensibilidad sin PageValues y ablacion de SpecialDay.

Guarda tablas en resultados/ y figuras en resultados/figuras/.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import (  # noqa: E402
    CSV,
    DIR_FIG,
    DIR_RES,
    ESPECIALES,
    SEMILLA,
    auditar,
    bootstrap_ic,
    cargar_datos,
    construir_modelos,
    contar_combinaciones,
    entrenar_familia,
    evaluar,
    matriz_confusion,
    particionar,
    predecir,
    puntuacion,
    resumen_particiones,
)
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def log(msg):
    print(msg, flush=True)


def ejecutar_escenario(nombre, X, y, itr, iva, ite, cv, incluir_pagevalues=True,
                       incluir_specialday=True, params_fijos=None):
    """Ajusta (o reutiliza) los cinco modelos y devuelve resultados y curvas."""
    modelos, mallas = construir_modelos(
        incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday
    )
    resultados = {}
    cv_todos = []
    for m in modelos:
        t0 = time.time()
        if params_fijos is not None:
            modelos[m].set_params(**params_fijos[m])
            modelos[m].fit(X.iloc[itr], y.iloc[itr])
            from proyecto_ml import ResultadoFamilia

            res = ResultadoFamilia(
                nombre=m,
                mejor_params=params_fijos[m],
                cv_ap=np.nan,
                cv_std=np.nan,
                modelo=modelos[m],
                tiempo=time.time() - t0,
            )
            cv_todos.append(pd.DataFrame())
            log(f"  [{nombre}] {m}: ajuste con hiperparametros fijos ({res.tiempo:.1f}s)")
        else:
            res, cv_df = entrenar_familia(m, modelos[m], mallas[m], X.iloc[itr], y.iloc[itr], cv)
            cv_todos.append(cv_df)
            log(
                f"  [{nombre}] {m}: AP_CV={res.cv_ap:.4f} +- {res.cv_std:.4f} "
                f"({res.tiempo:.1f}s) {res.mejor_params}"
            )
        resultados[m] = res
    cv_df = pd.concat([d for d in cv_todos if len(d)], ignore_index=True) if any(len(d) for d in cv_todos) else pd.DataFrame()
    return resultados, cv_df, mallas


def metricas_particiones(resultados, X, y, itr, iva, ite):
    filas = []
    for nombre, res in resultados.items():
        for part, ix in [("entrenamiento", itr), ("validacion", iva), ("test", ite)]:
            d = evaluar(res, X.iloc[ix], y.iloc[ix], part)
            filas.append(d)
    return pd.DataFrame(filas)


def curvas(resultados, X, y, ix, particion="validacion"):
    out = {}
    for nombre, res in resultados.items():
        ys = puntuacion(res.modelo, X.iloc[ix])
        yt = y.iloc[ix].to_numpy()
        pr_p, pr_r, _ = precision_recall_curve(yt, ys)
        fpr, tpr, _ = roc_curve(yt, ys)
        out[nombre] = {
            "pr_precision": pr_p.tolist(),
            "pr_recall": pr_r.tolist(),
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
        }
    return out


def main():
    DIR_RES.mkdir(parents=True, exist_ok=True)
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    t_inicio = time.time()

    log("Cargando y auditando datos...")
    df = cargar_datos(CSV)
    aud = auditar(df)
    (DIR_RES / "auditoria.json").write_text(
        json.dumps(aud, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df_u, X, y, itr, iva, ite = particionar(df)
    resumen_particiones(y, itr, iva, ite).to_csv(
        DIR_RES / "resumen_particiones.csv", index=False
    )
    log(resumen_particiones(y, itr, iva, ite).to_string(index=False))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    log("\n=== Escenario principal (con PageValues, con SpecialDay) ===")
    resultados, cv_df, mallas = ejecutar_escenario(
        "principal", X, y, itr, iva, ite, cv
    )
    cv_df.to_csv(DIR_RES / "cv_todos_los_candidatos.csv", index=False)

    comb = contar_combinaciones(mallas)
    comb.to_csv(DIR_RES / "combinaciones_malla.csv", index=False)
    log(comb.to_string(index=False))

    mejores = pd.DataFrame(
        [
            {
                "modelo": m,
                "mejores_hiperparametros": json.dumps(r.mejor_params, ensure_ascii=False),
                "AP_CV": r.cv_ap,
                "std_CV": r.cv_std,
                "tiempo_seg": r.tiempo,
            }
            for m, r in resultados.items()
        ]
    ).sort_values("AP_CV", ascending=False)
    mejores.to_csv(DIR_RES / "mejores_configuraciones.csv", index=False)
    log("\nMejores por familia:\n" + mejores.to_string(index=False))

    # Referencia trivial
    dummy = DummyClassifier(strategy="prior").fit(X.iloc[itr], y.iloc[itr])
    pos_rate = float(y.iloc[iva].mean())
    log(f"Referencia Dummy (clase mayoritaria): AP={pos_rate:.4f} en validacion")

    log("\n=== Metricas train / validacion / test ===")
    met = metricas_particiones(resultados, X, y, itr, iva, ite)
    met.to_csv(DIR_RES / "metricas_particiones.csv", index=False)

    # Ganador por AP de validacion
    val = met[met.particion == "validacion"].sort_values("AP", ascending=False)
    ganador = val.iloc[0]["modelo"]
    log(f"Ganador por AP de validacion: {ganador}")
    (DIR_RES / "ganador.json").write_text(
        json.dumps(
            {
                "modelo": ganador,
                "criterio": "mayor AP media de CV por familia; mayor AP en validacion entre finalistas",
                "AP_validacion": float(val.iloc[0]["AP"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("\n=== Curvas PR/ROC en validacion ===")
    cur = curvas(resultados, X, y, iva)
    (DIR_RES / "curvas_validacion.json").write_text(json.dumps(cur), encoding="utf-8")

    log("\n=== Matrices de confusion en test ===")
    mc = {}
    for m, r in resultados.items():
        cm = matriz_confusion(r, X.iloc[ite], y.iloc[ite])
        mc[m] = cm.tolist()
        log(f"{m}:\n{cm}")
    (DIR_RES / "matrices_confusion_test.json").write_text(json.dumps(mc), encoding="utf-8")

    log("\n=== Intervalos de confianza bootstrap (2000 replicas) ===")
    filas_ic = []
    for m, r in resultados.items():
        for part, ix in [("entrenamiento", itr), ("validacion", iva), ("test", ite)]:
            yt = y.iloc[ix].to_numpy()
            yp = predecir(r.modelo, X.iloc[ix])
            ys = puntuacion(r.modelo, X.iloc[ix])
            ic = bootstrap_ic(yt, yp, ys)
            ic["modelo"] = m
            ic["particion"] = part
            filas_ic.append(ic[["modelo", "particion", "metrica", "media_bootstrap", "ic_inferior", "ic_superior"]])
            log(f"  {m} / {part}: IC calculados")
    ic_df = pd.concat(filas_ic, ignore_index=True)
    ic_df.to_csv(DIR_RES / "metricas_intervalos_confianza.csv", index=False)

    # Equivalencia numerica vs ordinal de SpecialDay
    sd = X["SpecialDay"].to_numpy()
    ordinal = np.rint(sd / 0.2)
    escala = (sd - sd.mean()) / sd.std()
    escala_ord = (ordinal - ordinal.mean()) / ordinal.std()
    corr = float(np.corrcoef(escala, escala_ord)[0, 1])
    (DIR_RES / "specialday_equivalencia.json").write_text(
        json.dumps(
            {
                "valores_unicos": sorted(float(v) for v in np.unique(sd)),
                "correlacion_representaciones_escaladas": corr,
                "conclusion": "SpecialDay es equiespaciado; la codificacion ordinal es una transformacion afín de la numerica, por lo que tras estandarizar ambas representaciones son identicas.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"SpecialDay: correlacion numerica vs ordinal escalada = {corr:.6f}")

    log("\n=== Sensibilidad sin PageValues (reajuste completo) ===")
    res_sp, cv_sp, mallas_sp = ejecutar_escenario(
        "sin_pagevalues", X, y, itr, iva, ite, cv, incluir_pagevalues=False
    )
    cv_sp.to_csv(DIR_RES / "cv_sin_pagevalues.csv", index=False)
    met_sp = metricas_particiones(res_sp, X, y, itr, iva, ite)
    met_sp = met_sp[met_sp.particion.isin(["entrenamiento", "validacion"])]
    met_sp.to_csv(DIR_RES / "sensibilidad_sin_pagevalues.csv", index=False)
    comp = []
    for m in resultados:
        a = met[(met.modelo == m) & (met.particion == "validacion")].iloc[0]
        b = met_sp[(met_sp.modelo == m) & (met_sp.particion == "validacion")].iloc[0]
        comp.append(
            {
                "modelo": m,
                "AP_con_PageValues": a["AP"],
                "AP_sin_PageValues": b["AP"],
                "delta_AP": a["AP"] - b["AP"],
                "ROC_AUC_con": a["ROC_AUC"],
                "ROC_AUC_sin": b["ROC_AUC"],
            }
        )
    pd.DataFrame(comp).to_csv(DIR_RES / "comparacion_pagevalues.csv", index=False)
    log(pd.DataFrame(comp).to_string(index=False))

    log("\n=== Ablacion sin SpecialDay (hiperparametros fijos) ===")
    params = {m: r.mejor_params for m, r in resultados.items()}
    res_abl, _, _ = ejecutar_escenario(
        "sin_specialday", X, y, itr, iva, ite, cv,
        incluir_specialday=False, params_fijos=params,
    )
    abl = metricas_particiones(res_abl, X, y, itr, iva, ite)
    abl = abl[abl.particion.isin(["entrenamiento", "validacion"])]
    abl.to_csv(DIR_RES / "ablacion_sin_specialday.csv", index=False)
    comparacion_abl = []
    for m in resultados:
        a = met[(met.modelo == m) & (met.particion == "validacion")].iloc[0]
        b = abl[(abl.modelo == m) & (abl.particion == "validacion")].iloc[0]
        comparacion_abl.append(
            {
                "modelo": m,
                "AP_con_SpecialDay": a["AP"],
                "AP_sin_SpecialDay": b["AP"],
                "delta_AP": a["AP"] - b["AP"],
            }
        )
    pd.DataFrame(comparacion_abl).to_csv(
        DIR_RES / "comparacion_specialday.csv", index=False
    )
    log(pd.DataFrame(comparacion_abl).to_string(index=False))

    # Manifiesto
    manifiesto = {
        "sha256_csv": aud["sha256"],
        "semilla": SEMILLA,
        "python": aud["version_python"],
        "sklearn": aud["version_sklearn"],
        "n_train": int(len(itr)),
        "n_validacion": int(len(iva)),
        "n_test": int(len(ite)),
        "duplicados_eliminados": aud["duplicados_exactos"],
        "tiempo_total_seg": round(time.time() - t_inicio, 1),
    }
    (DIR_RES / "manifiesto.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nListo en {manifiesto['tiempo_total_seg']} s")


if __name__ == "__main__":
    main()

"""Genera las figuras de la seccion 1 del Entregable II."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import DIR_FIG, DIR_RES  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
MODELOS = ["RandomForest", "MLP", "SVM", "kNN", "Regresion_logistica"]
ETIQUETAS = {
    "RandomForest": "Random Forest",
    "MLP": "Red neuronal (MLP)",
    "SVM": "SVM (RBF)",
    "kNN": "k-NN",
    "Regresion_logistica": "Regresion logistica",
}
COLORES = {
    "RandomForest": "#2c7fb8",
    "MLP": "#d95f02",
    "SVM": "#7570b3",
    "kNN": "#1b9e77",
    "Regresion_logistica": "#e7298a",
}


def fig_distribucion():
    aud = json.loads((DIR_RES / "auditoria.json").read_text(encoding="utf-8"))
    clases = aud["clases"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    etq = ["Sin compra (0)", "Compra (1)"]
    vals = [clases["False"], clases["True"]]
    barras = ax.bar(etq, vals, color=["#9ecae1", "#3182bd"])
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v} ({v/sum(vals):.1%})",
                ha="center", fontsize=8)
    ax.set_ylabel("Numero de sesiones")
    ax.set_xlabel("Clase de Revenue")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Distribucion de clases")
    fig.savefig(DIR_FIG / "fig01_distribucion_clases.png")
    plt.close(fig)


def fig_mensual():
    mes = pd.read_csv(DIR_RES / "cobertura_mensual_train.csv")
    mes = mes.sort_values("tasa_compra", ascending=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(mes["Month"], mes["sesiones"], color="#9ecae1", label="Sesiones (train)")
    ax.set_xlabel("Mes de la sesion")
    ax.set_ylabel("Numero de sesiones (entrenamiento)")
    ax2 = ax.twinx()
    ax2.plot(mes["Month"], mes["tasa_compra"], "o-", color="#d95f02",
             label="Tasa de compra")
    ax2.set_ylabel("Tasa de compra")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title("Cobertura mensual y tasa de compra en entrenamiento")
    fig.savefig(DIR_FIG / "fig02_cobertura_mensual.png")
    plt.close(fig)


def _curva_param(ax, cv, modelo, param, xlabel, logx=False):
    sub = cv[cv.modelo == modelo].copy()
    sub["clave"] = sub[param].astype(str)
    agg = sub.groupby("clave", as_index=False).agg(
        ap=("mean_test_score", "mean"), std=("mean_test_score", "std"),
        train=("mean_train_score", "mean"))
    orden = sub.drop_duplicates("clave")[["clave", param]].copy()
    try:
        orden["v"] = orden[param].map(lambda v: float(str(v).replace("[", "").replace("]", "").split(",")[0])
                                      if str(v) not in ("sqrt", "0.5", "balanced", "None", "linear", "rbf", "uniform", "distance", "scale", "True", "False") else np.nan)
    except Exception:
        orden["v"] = np.nan
    if orden["v"].notna().all():
        orden = orden.sort_values("v")
        etiquetas = orden["clave"].tolist()
    else:
        orden = orden.sort_values("clave")
        etiquetas = orden["clave"].tolist()
    agg = agg.set_index("clave").loc[etiquetas].reset_index()
    x = np.arange(len(etiquetas))
    ax.errorbar(x, agg["ap"], yerr=agg["std"], marker="o", capsize=3,
                color=COLORES.get(modelo, "C0"), label="Validacion (AP media)")
    ax.plot(x, agg["train"], "s--", alpha=0.5, color="gray", label="Entrenamiento")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7, rotation=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AP")
    ax.legend(fontsize=7)


def fig_hiperparametros():
    cv = pd.read_csv(DIR_RES / "cv_todos_los_candidatos.csv")
    paneles = [
        ("Regresion_logistica", "param_clf__C", "C (regularizacion L2)"),
        ("kNN", "param_clf__n_neighbors", "Numero de vecinos (k)"),
        ("RandomForest", "param_clf__max_depth", "Profundidad maxima"),
        ("MLP", "param_clf__hidden_layer_sizes", "Neuronas por capa oculta"),
        ("SVM", "param_clf__C", "C (SVM)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    axes = axes.ravel()
    for ax, (m, p, xl) in zip(axes, paneles):
        _curva_param(ax, cv, m, p, xl)
        ax.set_title(ETIQUETAS[m], fontsize=9)
    # SVM: efecto del kernel
    ax = axes[5]
    sub = cv[cv.modelo == "SVM"].groupby("param_clf__kernel")["mean_test_score"].agg(["mean", "std"])
    ax.bar(sub.index, sub["mean"], yerr=sub["std"], capsize=4, color="#7570b3")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("AP (validacion)")
    ax.set_title("SVM: efecto del kernel", fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "fig03_efecto_hiperparametros.png")
    plt.close(fig)


def fig_train_vs_cv():
    met = pd.read_csv(DIR_RES / "metricas_particiones.csv")
    met = met[met.particion.isin(["entrenamiento", "validacion"])]
    piv = met.pivot_table(index="modelo", columns="particion", values="AP").loc[MODELOS]
    x = np.arange(len(MODELOS))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x - 0.2, piv["entrenamiento"], width=0.4, label="Entrenamiento", color="#bdbdbd")
    ax.bar(x + 0.2, piv["validacion"], width=0.4, label="Validacion", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=8, rotation=12)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Modelo")
    ax.legend()
    ax.set_title("Sobreajuste: AP en entrenamiento frente a validacion")
    fig.savefig(DIR_FIG / "fig04_train_vs_validacion.png")
    plt.close(fig)


def fig_curvas_pr_roc():
    cur = json.loads((DIR_RES / "curvas_validacion.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m in MODELOS:
        ax.plot(cur[m]["pr_recall"], cur[m]["pr_precision"], color=COLORES[m],
                label=f"{ETIQUETAS[m]}")
    ax.axhline(0.156, color="k", ls=":", linewidth=1, label="Tasa base (0,156)")
    ax.set_xlabel("Recall (clase compra)")
    ax.set_ylabel("Precision (clase compra)")
    ax.set_title("Curvas precision-recall en validacion")
    ax.legend(fontsize=7, loc="upper right")
    fig.savefig(DIR_FIG / "fig05_curvas_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m in MODELOS:
        ax.plot(cur[m]["roc_fpr"], cur[m]["roc_tpr"], color=COLORES[m],
                label=ETIQUETAS[m])
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC en validacion")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(DIR_FIG / "fig06_curvas_roc.png")
    plt.close(fig)


def fig_confusion():
    mc = json.loads((DIR_RES / "matrices_confusion_test.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.9))
    for ax, m in zip(axes, MODELOS):
        cm = np.array(mc[m])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Si"], fontsize=8)
        ax.set_yticklabels(["No", "Si"], fontsize=8)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        ax.set_title(ETIQUETAS[m], fontsize=8)
    fig.suptitle("Matrices de confusion en test (filas: clase real; columnas: prediccion)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_FIG / "fig07_matrices_confusion_test.png")
    plt.close(fig)


def fig_ap_ic():
    ic = pd.read_csv(DIR_RES / "metricas_intervalos_confianza.csv")
    ic = ic[(ic.metrica == "AP") & (ic.particion.isin(["validacion", "test"]))]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(MODELOS))
    ancho = 0.35
    for k, part in enumerate(["validacion", "test"]):
        sub = ic[ic.particion == part].set_index("modelo").loc[MODELOS]
        err = np.vstack([sub["media_bootstrap"] - sub["ic_inferior"],
                         sub["ic_superior"] - sub["media_bootstrap"]])
        ax.bar(x + (k - 0.5) * ancho, sub["media_bootstrap"], width=ancho,
               yerr=err, capsize=3, label=part.capitalize(),
               color=["#2c7fb8", "#fdae6b"][k])
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=8, rotation=12)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Modelo")
    ax.legend()
    ax.set_title("AP con intervalos de confianza bootstrap del 95 %")
    fig.savefig(DIR_FIG / "fig08_ap_intervalos.png")
    plt.close(fig)


def fig_pagevalues():
    cmp = pd.read_csv(DIR_RES / "comparacion_pagevalues.csv").set_index("modelo").loc[MODELOS]
    x = np.arange(len(MODELOS))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(x - 0.2, cmp["AP_con_PageValues"], width=0.4, label="Con PageValues", color="#2c7fb8")
    ax.bar(x + 0.2, cmp["AP_sin_PageValues"], width=0.4, label="Sin PageValues", color="#cb181d")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=8, rotation=12)
    ax.set_ylabel("AP en validacion")
    ax.set_xlabel("Modelo")
    ax.legend()
    ax.set_title("Sensibilidad al retirar PageValues")
    fig.savefig(DIR_FIG / "fig09_sensibilidad_pagevalues.png")
    plt.close(fig)


def fig_specialday():
    abl = pd.read_csv(DIR_RES / "comparacion_specialday.csv").set_index("modelo").loc[MODELOS]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(MODELOS))
    ax.bar(x, abl["delta_AP"], color="#756bb1")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=8, rotation=12)
    ax.set_ylabel("Cambio en AP (validacion)")
    ax.set_xlabel("Modelo")
    ax.set_title("Ablacion: efecto de retirar SpecialDay")
    fig.savefig(DIR_FIG / "fig10_ablacion_specialday.png")
    plt.close(fig)


def main():
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig_distribucion()
    fig_mensual()
    fig_hiperparametros()
    fig_train_vs_cv()
    fig_curvas_pr_roc()
    fig_confusion()
    fig_ap_ic()
    fig_pagevalues()
    fig_specialday()
    print("Figuras generadas en", DIR_FIG)


if __name__ == "__main__":
    main()

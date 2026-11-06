"""Auditoria del conjunto de datos y generacion de particiones."""

import json

import pandas as pd

from proyecto_ml import (
    CSV,
    DIR_RES,
    cargar_datos,
    auditar,
    tabla_auditoria,
    particionar,
    resumen_particiones,
)


def main():
    DIR_RES.mkdir(parents=True, exist_ok=True)
    df = cargar_datos(CSV)
    aud = auditar(df)
    (DIR_RES / "auditoria.json").write_text(
        json.dumps(aud, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tabla_auditoria(df).to_csv(DIR_RES / "tabla_variables.csv", index=False)

    df_u, X, y, itr, iva, ite = particionar(df)
    resumen_particiones(y, itr, iva, ite).to_csv(
        DIR_RES / "resumen_particiones.csv", index=False
    )
    pd.DataFrame(
        {
            "fila_original_sin_duplicados": range(len(df_u)),
            "particion": ["entrenamiento"] * len(itr)
            + ["validacion"] * len(iva)
            + ["test"] * len(ite),
            "indice": list(itr) + list(iva) + list(ite),
        }
    ).to_csv(DIR_RES / "particiones.csv", index=False)

    # Cobertura mensual y tasa de compra por mes (solo entrenamiento)
    mes = df_u.iloc[itr].groupby("Month", observed=True)["Revenue"].agg(
        ["size", "mean"]
    )
    mes.columns = ["sesiones", "tasa_compra"]
    mes.to_csv(DIR_RES / "cobertura_mensual_train.csv")

    print("Auditoria:")
    print(json.dumps({k: aud[k] for k in [
        "filas", "columnas", "nulos_totales", "duplicados_exactos",
        "clases", "proporcion_compra", "valores_especialday"]}, ensure_ascii=False, indent=2))
    print("\nParticiones:")
    print(resumen_particiones(y, itr, iva, ite).to_string(index=False))
    print("\nCobertura mensual (train):")
    print(mes.to_string())


if __name__ == "__main__":
    main()

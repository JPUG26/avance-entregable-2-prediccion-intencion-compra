# Predicción de la intención de compra de usuarios en línea mediante aprendizaje supervisado

**Proyecto de aula — Modelos y Simulación de Sistemas II**
Departamento de Ingeniería de Sistemas, Universidad de Antioquia, Colombia

**Integrantes:** Mateo Upegui Borja, Juan P. Uribe González y Juan D. Rincón Bedoya.

## Descripción

Clasificación binaria supervisada de la intención de compra sobre el *Online Shoppers
Purchasing Intention Dataset* (UCI #468, Sakar y Kastro, 2018): 12 330 sesiones de navegación,
17 atributos predictores y la etiqueta `Revenue` (15,5 % de compras). Este repositorio contiene
el **Entregable II**, sección 1 (30 %): configuración experimental y resultados del entrenamiento
y la evaluación de cinco familias de modelos.

## Estructura del repositorio

```
.
├── clipboard_1790819347953_online_shoppers_intention.csv   # datos de entrada
├── informe_entregable_II.docx / .pdf                       # informe (plantilla IEEE)
├── notebooks/
│   └── 02_entrenamiento_evaluacion.ipynb                   # notebook reproducible
├── src/
│   ├── proyecto_ml.py     # utilidades compartidas (pipelines, métricas, IC)
│   ├── auditar.py         # auditoría y particiones
│   ├── entrenar.py        # búsqueda, evaluación y análisis
│   ├── figuras.py         # gráficas del informe
│   ├── informe.py         # genera DOCX y PDF
│   └── crear_notebook.py  # construye el notebook
├── resultados/            # tablas CSV, figuras y manifiesto
└── requirements.txt
```

## Cómo reproducir

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Opción A — notebook (recomendada):

1. Abra `notebooks/02_entrenamiento_evaluacion.ipynb`.
2. Ejecute las celdas en orden. Con `RECALCULAR = False` se cargan los resultados guardados;
   cambie a `True` para reentrenar todo desde el CSV (≈12 minutos).

Opción B — línea de comandos:

```bash
python src/auditar.py     # auditoría y particiones
python src/entrenar.py    # entrenamiento, evaluación y análisis
python src/figuras.py     # figuras
python src/informe.py     # informe DOCX y PDF
```

## Diseño experimental (resumen)

- Partición estratificada 60/20/20 (entrenamiento, validación, prueba); el test se mantiene
  aislado hasta la evaluación final.
- Preprocesamiento dentro de los folds con `Pipeline` + `ColumnTransformer`:
  `log1p` opcional + estandarización, one-hot para categóricas, imputación ajustada en train.
- Búsqueda en rejilla con validación cruzada estratificada de 5 folds, `scoring=average_precision`,
  con la comparación `log1p` como hiperparámetro.
- Cinco familias: regresión logística, k-NN, Random Forest, red neuronal (MLP) y SVM (RBF).
- Métricas: AP (precisión-recall), ROC-AUC, F1, precisión, recall y exactitud balanceada, con
  intervalos de confianza del 95 % por bootstrap estratificado (2000 réplicas).
- Análisis de sensibilidad sin `PageValues` y ablación de `SpecialDay`.

## Datos y licencia

*Online Shoppers Purchasing Intention Dataset*, C. Sakar y Y. Kastro, UCI Machine Learning
Repository, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licencia CC BY 4.0.

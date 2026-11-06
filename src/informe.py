"""Genera el informe del Entregable II (seccion 1: 30 %) en formato IEEE.

Construye el DOCX a partir de los resultados guardados en resultados/ y lo
convierte a PDF con Word mediante automatizacion COM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
FIG = RES / "figuras"
SALIDA_DOCX = RAIZ / "informe_entregable_II.docx"
SALIDA_PDF = RAIZ / "informe_entregable_II.pdf"

ETQ = {
    "RandomForest": "Random Forest",
    "MLP": "Red neuronal (MLP)",
    "SVM": "SVM (RBF)",
    "kNN": "k-NN",
    "Regresion_logistica": "Regresion logistica",
}
ORDEN = ["RandomForest", "MLP", "SVM", "kNN", "Regresion_logistica"]


def num(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def cargar():
    d = {}
    d["aud"] = json.loads((RES / "auditoria.json").read_text(encoding="utf-8"))
    d["part"] = pd.read_csv(RES / "resumen_particiones.csv")
    d["comb"] = pd.read_csv(RES / "combinaciones_malla.csv")
    d["mej"] = pd.read_csv(RES / "mejores_configuraciones.csv")
    d["met"] = pd.read_csv(RES / "metricas_particiones.csv")
    d["ic"] = pd.read_csv(RES / "metricas_intervalos_confianza.csv")
    d["cmp_pv"] = pd.read_csv(RES / "comparacion_pagevalues.csv")
    d["cmp_sd"] = pd.read_csv(RES / "comparacion_specialday.csv")
    d["gan"] = json.loads((RES / "ganador.json").read_text(encoding="utf-8"))
    d["sd"] = json.loads((RES / "specialday_equivalencia.json").read_text(encoding="utf-8"))
    d["mes"] = pd.read_csv(RES / "cobertura_mensual_train.csv")
    d["tab"] = pd.read_csv(RES / "tabla_variables.csv")
    return d


def configurar_estilo(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(3)
    st.paragraph_format.line_spacing = 1.0
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def dos_columnas(section):
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "300")


def par(doc, texto, size=9.5, italic=False, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        space_after=3):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(texto)
    r.font.size = Pt(size)
    r.italic = italic
    r.bold = bold
    return p


def titulo_seccion(doc, texto, numero=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"{numero}. {texto}" if numero else texto)
    r.bold = True
    r.font.size = Pt(10)
    r.font.name = "Times New Roman"
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def subtitulo(doc, texto):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(texto)
    r.italic = True
    r.bold = True
    r.font.size = Pt(9.5)
    return p


def caption(doc, texto):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(texto)
    r.font.size = Pt(8)
    r.italic = True
    return p


def figura(doc, archivo, texto_caption, ancho=3.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.add_run().add_picture(str(FIG / archivo), width=Inches(ancho))
    caption(doc, texto_caption)


def tabla(doc, encabezados, filas, texto_caption, anchos=None, size=7.5):
    t = doc.add_table(rows=1, cols=len(encabezados))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(encabezados):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(size)
    for fila in filas:
        celdas = t.add_row().cells
        for i, v in enumerate(fila):
            celdas[i].text = ""
            r = celdas[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
    caption(doc, texto_caption)
    return t


def main():
    d = cargar()
    aud, met, ic, gan = d["aud"], d["met"], d["ic"], d["gan"]
    doc = Document()
    configurar_estilo(doc)
    s0 = doc.sections[0]
    s0.top_margin = Inches(0.75)
    s0.bottom_margin = Inches(0.75)
    s0.left_margin = Inches(0.7)
    s0.right_margin = Inches(0.7)

    # ------------------------- Cabecera (una columna) -------------------------
    p = doc.add_paragraph()
    r = p.add_run("Predicción de la intención de compra de usuarios en línea "
                  "mediante aprendizaje supervisado")
    r.bold = True
    r.font.size = Pt(17)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    par(doc, "Mateo Upegui Borja, Juan P. Uribe González y Juan D. Rincón Bedoya",
        size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    par(doc, "Modelos y Simulación de Sistemas II, Departamento de Ingeniería de "
             "Sistemas, Universidad de Antioquia, Medellín, Colombia",
        size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Resumen—")
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run(
        "Este documento corresponde al Entregable II e incorpora los apartados del "
        "Entregable I con las correcciones solicitadas, junto con el diseño experimental "
        "y los resultados del entrenamiento y la evaluación de modelos de aprendizaje "
        "automático. Se aborda la predicción de la intención de compra como un problema "
        "de clasificación binaria supervisada sobre el conjunto Online Shoppers Purchasing "
        f"Intention (UCI), con {aud['filas']} sesiones, {aud['columnas']} columnas y una "
        f"prevalencia de compra del {num(aud['proporcion_compra']*100,2)} %. Se comparan "
        "cinco familias de modelos: regresión logística, k-NN, Random Forest, una red "
        "neuronal (MLP) y una máquina de vectores de soporte. El preprocesamiento y la "
        "selección de transformaciones se ajustan exclusivamente con datos de "
        "entrenamiento mediante validación cruzada estratificada, y el conjunto de prueba "
        "se mantiene aislado hasta la evaluación final. La métrica principal es el área "
        "bajo la curva precisión-recall (AP), acompañada de ROC-AUC, F1 y exactitud "
        "balanceada con intervalos de confianza por bootstrap. Random Forest obtiene el "
        f"mejor desempeño, con AP de {num(gan['AP_validacion'])} en validación. Se analiza "
        "además la sensibilidad al retirar PageValues, cuya contribución resulta dominante, "
        "y se verifica que el tratamiento numérico y el ordinal de SpecialDay son "
        "equivalentes para este conjunto de datos."
    )
    r2.font.size = Pt(9)

    p = doc.add_paragraph()
    r = p.add_run("Términos de indexación—")
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run("Intención de compra, clasificación desbalanceada, comercio "
                   "electrónico, selección de hiperparámetros, validación cruzada "
                   "estratificada, intervalos de confianza.")
    r2.font.size = Pt(9)

    # ------------------------- Cuerpo a dos columnas -------------------------
    doc.add_section(WD_SECTION.CONTINUOUS)
    dos_columnas(doc.sections[-1])

    # I. Introducción
    titulo_seccion(doc, "Introducción", "I")
    par(doc,
        "El comercio electrónico se ha consolidado como uno de los canales de venta de "
        "mayor crecimiento. Sin embargo, su tasa de conversión, entendida como la "
        "proporción de visitas que terminan en una transacción, es baja: la mayoría de las "
        "sesiones concluye sin compra. A diferencia de una tienda física, donde un vendedor "
        "puede observar el comportamiento de un cliente y adaptar su atención, en un "
        "entorno virtual esa capacidad de reacción no existe de forma natural. Esta "
        "carencia ha motivado el desarrollo de sistemas que, a partir del rastro de "
        "navegación, estimen la probabilidad de que una sesión termine en compra, de modo "
        "que la plataforma pueda intervenir mediante ofertas personalizadas, asistencia en "
        "línea o ajustes en el contenido mostrado [1], [2].")
    par(doc,
        "Predecir la intención de compra a partir de datos de navegación es un problema "
        "apropiado para el aprendizaje automático por tres razones. Primero, la relación "
        "entre las métricas de comportamiento y la decisión final del usuario es compleja "
        "y no lineal, por lo que resulta difícil de capturar con reglas manuales. Segundo, "
        "las plataformas generan de forma continua registros históricos etiquetados, lo que "
        "habilita el aprendizaje supervisado. Tercero, la decisión debe tomarse de manera "
        "automática y escalable sobre miles de sesiones.")
    par(doc,
        "El primer entregable describió el problema, caracterizó la base de datos y definió "
        "el paradigma y el protocolo de validación. Este segundo entregable desarrolla la "
        "fase de entrenamiento y evaluación: se especifica el diseño experimental, se "
        "comparan cinco familias de modelos con búsqueda de hiperparámetros, se reportan "
        "resultados de entrenamiento, validación y prueba con intervalos de confianza y se "
        "discuten las implicaciones de los hallazgos.")

    # II. Descripción del problema
    titulo_seccion(doc, "Descripción del problema", "II")
    subtitulo(doc, "A. Contexto y utilidad del aprendizaje automático")
    par(doc,
        "En una tienda física el vendedor identifica al cliente con intención de compra y "
        "lo orienta; en línea esa interacción no existe de forma natural. Estimar la "
        "probabilidad de compra de una sesión permite actuar sobre ella y priorizar "
        "recursos: ofrecer un cupón al usuario indeciso, abrir un chat de asistencia o "
        "personalizar recomendaciones. El problema es adecuado para el aprendizaje "
        "automático porque la relación entre las métricas de navegación, como las páginas "
        "vistas, los tiempos de permanencia, las tasas de rebote y de salida y el valor de "
        "página, y la compra no es lineal ni expresable mediante reglas manuales; porque "
        "existen datos históricos etiquetados; y porque la decisión debe tomarse "
        "automáticamente sobre miles de sesiones. Los datos fueron publicados por Sakar y "
        "Kastro en el repositorio UCI [1].")
    subtitulo(doc, "B. Composición de la base de datos")
    par(doc,
        f"El conjunto Online Shoppers Purchasing Intention contiene {aud['filas']} sesiones "
        f"descritas por {aud['columnas']} columnas: 17 atributos predictores y la etiqueta "
        f"binaria Revenue. La prevalencia de compra es del {num(aud['proporcion_compra']*100,2)} % "
        f"({aud['clases']['True']} sesiones con compra frente a {aud['clases']['False']} sin "
        "compra), lo que confirma un desbalance marcado. La auditoría realizada encontró "
        f"{aud['nulos_totales']} valores faltantes y {aud['duplicados_exactos']} filas "
        "duplicadas exactas, que se eliminaron antes de particionar para evitar que una "
        "misma sesión apareciera simultáneamente en entrenamiento y prueba. Los atributos "
        "incluyen variables de conteo, duraciones, tasas de comportamiento, el valor de "
        "página PageValues, la cercanía a fechas especiales SpecialDay, el mes, códigos "
        "enteros que representan categorías (sistema operativo, navegador, región y fuente "
        "de tráfico), el tipo de visitante y un indicador de fin de semana. La Tabla I "
        "detalla el tipo, el significado y la codificación de cada variable; la descripción "
        "de la codificación se amplía en el apartado de métodos, como se solicitó en la "
        "realimentación.")
    filas_tab = []
    tipo = {"numerica": "Numérica", "categorica": "Categórica"}
    cod = {
        "Administrative": "log1p + estandarizar",
        "Administrative_Duration": "log1p + estandarizar",
        "Informational": "log1p + estandarizar",
        "Informational_Duration": "log1p + estandarizar",
        "ProductRelated": "log1p + estandarizar",
        "ProductRelated_Duration": "log1p + estandarizar",
        "BounceRates": "log1p + estandarizar",
        "ExitRates": "log1p + estandarizar",
        "PageValues": "log1p + estandarizar",
        "SpecialDay": "Numérica estandarizada (ordinal equivalente)",
        "Month": "One-hot",
        "OperatingSystems": "One-hot",
        "Browser": "One-hot",
        "Region": "One-hot",
        "TrafficType": "One-hot",
        "VisitorType": "One-hot",
        "Weekend": "0/1 (sin cambios)",
        "Revenue": "0/1 (objetivo)",
    }
    for _, row in d["tab"].iterrows():
        v = row["variable"]
        filas_tab.append([v, tipo.get(row["tipo"], row["tipo"]), cod.get(v, "")])
    tabla(doc, ["Variable", "Tipo", "Codificación"], filas_tab,
          "Tabla I. Descripción y codificación de las variables del conjunto de datos.",
          size=7)
    figura(doc, "fig01_distribucion_clases.png",
           "Fig. 1. Distribución de clases de la variable objetivo Revenue.", ancho=2.9)
    figura(doc, "fig02_cobertura_mensual.png",
           "Fig. 2. Cobertura mensual y tasa de compra en el conjunto de entrenamiento.",
           ancho=3.3)
    subtitulo(doc, "C. Paradigma de aprendizaje")
    par(doc,
        "Se adopta el aprendizaje supervisado porque todas las sesiones cuentan con una "
        "etiqueta observada que indica si terminaron en compra, y la tarea es de "
        "clasificación binaria con clases desbalanceadas. El aprendizaje no supervisado "
        "podría emplearse como análisis complementario, por ejemplo para agrupar perfiles "
        "de navegación, pero no permitiría aprender a predecir la compra a partir de la "
        "etiqueta. Con un 84,5 % de clase negativa, un clasificador trivial que siempre "
        "prediga la clase mayoritaria alcanza una exactitud elevada sin aportar valor, lo "
        "que obliga a emplear métricas sensibles a la clase minoritaria y a justificar el "
        "protocolo de validación.")

    # III. Estado del arte
    titulo_seccion(doc, "Estado del arte", "III")
    par(doc,
        "La literatura que aborda la predicción de la intención de compra con la misma "
        "base de datos permite situar los resultados de este trabajo. Sakar et al. [2] "
        "comparan Random Forest, máquinas de vectores de soporte y perceptrones multicapa, "
        "aplican sobremuestreo para equilibrar las clases y seleccionan variables mediante "
        "correlación, información mutua y mRMR, un criterio de mínima redundancia y máxima "
        "relevancia que prefiere atributos muy relacionados con la clase y poco redundantes "
        "entre sí. Su protocolo repite cien veces una partición 70/30 y aparta el conjunto "
        "de prueba antes del balanceo. Necula [3] estudia el tiempo dedicado a consultar "
        "información de productos y combina análisis exploratorio con K-means y componentes "
        "principales; compara regresión logística, árboles de decisión, Random Forest y SVM, "
        "reserva el 25 % para prueba, aplica ADASYN, un sobremuestreo adaptativo de la clase "
        "minoritaria, y emplea validación cruzada estratificada de diez particiones. Satu e "
        "Islam [4] exploran la ingeniería de características con CFS, que busca subconjuntos "
        "de atributos correlacionados con la clase y poco redundantes, junto con "
        "transformaciones de escala y SMOTE, e informan el mejor resultado con Random "
        "Forest. Lin [5] compara SVM, XGBoost, CatBoost y una red neuronal, con codificación "
        "de categorías, detección de atípicos mediante Isolation Forest, SMOTE y selección "
        "de variables, y reporta el mejor desempeño con CatBoost.")
    tabla(
        doc,
        ["Trabajo", "Modelos", "Balanceo", "Validación", "Métricas y resultado"],
        [
            ["Sakar et al. (2019) [2]", "RF, SVM, MLP", "Sobremuestreo",
             "100 repeticiones 70/30", "Exactitud y F1; MLP superior"],
            ["Necula (2023) [3]", "RL, árboles, RF, SVM", "ADASYN",
             "CV estratificada 10 folds y prueba 25 %", "Exactitud, F1, ROC-AUC; RF y SVM"],
            ["Satu e Islam (2023) [4]", "RF, SVM, árboles, RL", "SMOTE",
             "CV de 10 particiones", "Exactitud, F1, ROC-AUC; RF estable"],
            ["Lin (2025) [5]", "SVM, XGBoost, CatBoost, red", "SMOTE",
             "80/20 con 50 semillas", "Precisión, recall, ROC-AUC; CatBoost"],
        ],
        "Tabla II. Comparación de los trabajos revisados sobre la misma base de datos.",
        size=7,
    )
    par(doc,
        "Esa revisión deja tres lecciones para el diseño experimental. Primero, como cada "
        "estudio preprocesa, balancea y valida de forma distinta, sus cifras no son "
        "directamente comparables y varias publicaciones no aclaran si el balanceo se "
        "ajustó solo con datos de entrenamiento. Por ello, en este trabajo el "
        "preprocesamiento y cualquier decisión de transformación se ajustan dentro de los "
        "folds, y el conjunto de prueba se aísla por completo. Segundo, el desbalance "
        "justifica privilegiar métricas como F1, ROC-AUC y el área bajo la curva "
        "precisión-recall, en lugar de la exactitud. Tercero, la comparación de las cinco "
        "familias de modelos, la exploración de la transformación logarítmica y la "
        "verificación del tratamiento de SpecialDay retoman directamente las técnicas y "
        "advertencias de estos antecedentes. Se advierte que las cifras exactas de los "
        "trabajos citados provienen del Entregable I y que el F1 atribuido a Sakar et al. "
        "está pendiente de confirmación en el artículo original, pues las fuentes "
        "secundarias difieren.")

    # IV. Configuración experimental
    titulo_seccion(doc, "Configuración experimental", "IV")
    subtitulo(doc, "A. Metodología de validación y preprocesamiento")
    par(doc,
        "El conjunto depurado se dividió de forma estratificada en tres particiones "
        "disjuntas: 60 % para entrenamiento, 20 % para validación y 20 % para prueba, con "
        f"semilla fija. La Tabla III resume su tamaño y balance. El ajuste de "
        "hiperparámetros se realizó con validación cruzada estratificada de cinco folds "
        "sobre el 60 % de entrenamiento, de modo que cada fold conserva la proporción de "
        "compra. El 20 % de validación se empleó para comparar los finalistas y elegir el "
        "ganador, y el 20 % de prueba se mantuvo aislado hasta la evaluación final, sin "
        "participar en la selección de modelos, transformaciones ni hiperparámetros. No se "
        "aplicó sub ni sobremuestreo en el experimento principal: se documenta el desbalance "
        "y se exploran pesos de clase balanceados, una alternativa que no genera datos "
        "sintéticos y que se ajusta dentro de cada fold. Todo el preprocesamiento se "
        "encapsuló en un Pipeline y un ColumnTransformer, de modo que la imputación, la "
        "transformación logarítmica y el escalado se aprenden únicamente con los datos de "
        "entrenamiento de cada fold.")
    filas_part = []
    for _, r in d["part"].iterrows():
        filas_part.append([
            r["particion"], int(r["n"]), int(r["positivos"]), int(r["negativos"]),
            num(r["proporcion_compra"], 4),
        ])
    tabla(doc, ["Partición", "n", "Compra", "Sin compra", "Proporción"],
          filas_part,
          "Tabla III. Tamaño y balance de las particiones tras eliminar duplicados.",
          size=8)
    par(doc,
        "La codificación se define por tipo de variable. Los nueve atributos numéricos de "
        "conteo, duración y tasa, incluido PageValues, se transforman con log1p, que admite "
        "valores iguales a cero sin indefinirse, y luego se estandarizan; la decisión de "
        "aplicar log1p se trató como un hiperparámetro más y se eligió con validación "
        "cruzada. Los códigos enteros de sistema operativo, navegador, región y fuente de "
        "tráfico, así como el mes y el tipo de visitante, se codifican con one-hot porque "
        "representan categorías y no magnitudes. Weekend se usa como indicador binario. "
        "SpecialDay, que toma valores equiespaciados entre cero y uno, se trató como "
        "numérica estandarizada.")
    subtitulo(doc, "B. Hiperparámetros y mallas de valores")
    par(doc,
        "Cada familia se ajustó con búsqueda exhaustiva en rejilla. La Tabla IV especifica "
        "los hiperparámetros analizados y la malla de valores. La comparación de "
        "preprocesamiento con y sin log1p se incluyó como un parámetro de la malla en los "
        "cinco modelos.")
    filas_hp = [
        ["Regresión logística", "C", "0,01; 0,1; 1; 10"],
        ["", "class_weight", "ninguno; balanceado"],
        ["", "log1p", "sí; no"],
        ["k-NN", "n_neighbors", "5; 15; 31"],
        ["", "weights", "uniforme; por distancia"],
        ["", "p", "1 (Manhattan); 2 (Euclídea)"],
        ["", "log1p", "sí; no"],
        ["Random Forest", "n_estimators", "300 (fijo)"],
        ["", "max_depth", "sin límite; 12"],
        ["", "min_samples_leaf", "1; 5"],
        ["", "max_features", "sqrt; 0,5"],
        ["", "class_weight", "ninguno; balanceado"],
        ["", "log1p", "sí; no"],
        ["Red neuronal (MLP)", "hidden_layer_sizes", "(32); (64); (64, 32)"],
        ["", "alpha", "0,0001; 0,01"],
        ["", "learning_rate_init", "0,001; 0,01"],
        ["", "log1p", "sí; no"],
        ["SVM", "kernel", "lineal; RBF"],
        ["", "C", "0,1; 1; 10"],
        ["", "gamma", "scale; 0,01"],
        ["", "class_weight", "ninguno; balanceado"],
        ["", "log1p", "sí; no"],
    ]
    tabla(doc, ["Modelo", "Hiperparámetro", "Valores analizados"], filas_hp,
          "Tabla IV. Hiperparámetros y mallas evaluadas en la búsqueda de cada modelo.",
          size=7)
    comp_txt = ", ".join(
        f"{ETQ[r['modelo']]} {int(r['combinaciones'])}"
        for _, r in d["comb"].iterrows()
    )
    par(doc,
        f"El número de combinaciones evaluadas por familia fue: {comp_txt}. Todos los "
        "modelos compartieron los mismos folds, lo que hace comparables sus puntuaciones "
        "de validación cruzada.")
    subtitulo(doc, "C. Métricas de desempeño")
    par(doc,
        "Con una prevalencia de compra cercana al 15,6 %, la exactitud es engañosa, pues un "
        "clasificador que siempre prediga la clase mayoritaria supera el 84 %. Por ello, la "
        "métrica principal de selección es el área bajo la curva precisión-recall, denotada "
        "AP y calculada como el promedio de la precisión en los distintos umbrales de "
        "recall. Se complementa con el ROC-AUC, que mide la capacidad de ordenar las "
        "sesiones por probabilidad de compra; con el F1, la precisión y el recall de la "
        "clase positiva, que resumen el compromiso entre falsos positivos y falsos "
        "negativos; y con la exactitud balanceada, que promedia el desempeño en ambas "
        "clases. Todas las medidas se reportaron con intervalos de confianza del 95 % "
        "estimados mediante bootstrap estratificado con dos mil réplicas, y se incluyeron "
        "resultados de entrenamiento, validación y prueba.")

    # V. Resultados
    titulo_seccion(doc, "Resultados del entrenamiento de modelos", "V")
    subtitulo(doc, "A. Efecto de los hiperparámetros")
    par(doc,
        "La Fig. 3 muestra el efecto de los hiperparámetros más relevantes sobre el AP de "
        "validación. Agrupar las combinaciones por el valor del parámetro, promediando el "
        "resto, permite leer la tendencia general. En la regresión logística la "
        "regularización intermedia resultó preferible a los extremos; en k-NN los valores "
        "grandes de vecinos redujeron el sobreajuste; en Random Forest una profundidad "
        "limitada controló la varianza; en la red neuronal una capa oculta de tamaño "
        "moderado fue suficiente; y en SVM el kernel RBF superó al lineal. En todos los "
        "casos la transformación log1p tendió a mejorar el desempeño de los modelos "
        "sensibles a la escala.")
    figura(doc, "fig03_efecto_hiperparametros.png",
           "Fig. 3. Efecto de los hiperparámetros sobre el AP de validación. Las barras de "
           "error corresponden a la desviación entre folds.")
    figura(doc, "fig04_train_vs_validacion.png",
           "Fig. 4. AP en entrenamiento frente a validación, que evidencia el grado de "
           "sobreajuste de cada familia.")
    subtitulo(doc, "B. Selección y desempeño de los modelos")
    par(doc,
        "La Tabla V resume la mejor configuración de cada familia según el AP medio de la "
        "validación cruzada. Random Forest obtuvo el mayor AP de validación cruzada "
        f"({num(float(d['mej'].iloc[0]['AP_CV']))}), seguido de la red neuronal, SVM, k-NN y la regresión logística. "
        "La Fig. 4 evidencia que k-NN y Random Forest ajustan casi perfectamente el "
        "entrenamiento, mientras que la regresión logística generaliza de forma más "
        "estable; en consecuencia, las diferencias deben leerse en validación y prueba, no "
        "en entrenamiento.")
    filas_best = []
    for _, r in d["mej"].iterrows():
        try:
            p = json.loads(r["mejores_hiperparametros"]) if isinstance(r["mejores_hiperparametros"], str) else r["mejores_hiperparametros"]
        except Exception:
            p = {}
        filas_best.append([
            ETQ.get(r["modelo"], r["modelo"]),
            num(r["AP_CV"]), num(r["std_CV"]),
            r["mejores_hiperparametros"].replace('"', "").replace("{", "").replace("}", ""),
        ])
    tabla(doc, ["Modelo", "AP CV", "Desv.", "Mejores hiperparámetros"],
          filas_best,
          "Tabla V. Mejor configuración por familia y AP medio de validación cruzada.",
          size=6.5)
    par(doc,
        "La Tabla VI presenta el desempeño en entrenamiento, validación y prueba con "
        "intervalos de confianza del 95 % para el AP. En validación, Random Forest alcanzó "
        f"el mayor AP ({num(float(met[(met.modelo=='RandomForest')&(met.particion=='validacion')].iloc[0]['AP']))}), "
        "y conservó el liderazgo en prueba, con un AP cercano al valor de validación, lo "
        "que indica que la selección no dependió de una partición afortunada. La brecha "
        "entre el AP de entrenamiento y el de prueba en Random Forest y k-NN es amplia y "
        "confirma su sobreajuste, mientras que la regresión logística muestra la menor "
        "brecha. El punto de referencia trivial, que siempre predice la clase mayoritaria, "
        "obtiene un AP igual a la prevalencia, cercano a 0,156, muy por debajo de todos los "
        "modelos.")
    filas_res = []
    for m in ORDEN:
        sub = met[met.modelo == m].set_index("particion")
        icm = ic[(ic.modelo == m) & (ic.metrica == "AP")].set_index("particion")

        def celda(part):
            ap = float(sub.loc[part, "AP"])
            if part in icm.index:
                lo = float(icm.loc[part, "ic_inferior"])
                hi = float(icm.loc[part, "ic_superior"])
                return f"{num(ap)} [{num(lo,3)}, {num(hi,3)}]"
            return num(ap)

        filas_res.append([
            ETQ[m],
            celda("entrenamiento"),
            celda("validacion"),
            celda("test"),
            num(float(sub.loc["test", "F1"])),
            num(float(sub.loc["test", "ROC_AUC"])),
            num(float(sub.loc["test", "balanced_accuracy"])),
        ])
    tabla(doc, ["Modelo", "AP train", "AP val", "AP test [IC 95 %]", "F1 test", "ROC-AUC test", "Bal. acc"],
          filas_res,
          "Tabla VI. Desempeño por partición. Los intervalos corresponden al AP en prueba.",
          size=6.5)
    figura(doc, "fig08_ap_intervalos.png",
           "Fig. 5. AP en validación y prueba con intervalos de confianza bootstrap del 95 %.")
    par(doc,
        "Las Figs. 6 y 7 muestran las curvas precisión-recall y ROC en validación. La curva "
        "precisión-recall de Random Forest domina en la mayor parte del rango de recall, "
        "mientras que la red neuronal y SVM presentan un comportamiento intermedio y la "
        "regresión logística queda rezagada. En ROC las diferencias son menores porque el "
        "desbalance afecta menos a esa curva, lo que refuerza la decisión de usar AP como "
        "métrica principal.")
    figura(doc, "fig05_curvas_pr.png",
           "Fig. 6. Curvas precisión-recall en validación para los cinco modelos.")
    figura(doc, "fig06_curvas_roc.png",
           "Fig. 7. Curvas ROC en validación para los cinco modelos.")
    par(doc,
        "La Fig. 8 presenta las matrices de confusión en prueba. Con el umbral nativo, "
        "Random Forest privilegia la precisión de la clase positiva, con pocos falsos "
        "positivos y más falsos negativos, mientras que la red neuronal y SVM recuperan más "
        "compra a costa de una menor precisión. La elección entre ambos perfiles depende "
        "del costo relativo de cada tipo de error en la aplicación.")
    figura(doc, "fig07_matrices_confusion_test.png",
           "Fig. 8. Matrices de confusión en el conjunto de prueba.", ancho=3.4)
    subtitulo(doc, "C. Sensibilidad y ablaciones")
    par(doc,
        "El análisis de sensibilidad reentrenó las cinco familias sin PageValues, "
        "manteniendo el mismo protocolo. La Fig. 9 y la Tabla VII muestran una caída "
        "drástica del AP, de más de treinta puntos porcentuales, lo que confirma que "
        "PageValues concentra buena parte de la señal predictiva disponible en la base. "
        "Este resultado es relevante porque PageValues se calcula a partir del "
        "comportamiento de la sesión completa, de modo que su disponibilidad antes de la "
        "compra es cuestionable; sin ella, el problema es notablemente más difícil.")
    figura(doc, "fig09_sensibilidad_pagevalues.png",
           "Fig. 9. AP en validación con y sin PageValues para cada familia.")
    filas_pv = []
    for _, r in d["cmp_pv"].iterrows():
        filas_pv.append([
            ETQ.get(r["modelo"], r["modelo"]), num(r["AP_con_PageValues"], 3),
            num(r["AP_sin_PageValues"], 3), num(r["delta_AP"], 3),
        ])
    tabla(doc, ["Modelo", "AP con PageValues", "AP sin PageValues", "Diferencia"],
          filas_pv,
          "Tabla VII. Sensibilidad al retirar PageValues (AP en validación).", size=7)
    corr = d["sd"]["correlacion_representaciones_escaladas"]
    filas_sd = []
    for _, r in d["cmp_sd"].iterrows():
        filas_sd.append([
            ETQ.get(r["modelo"], r["modelo"]),
            num(r["AP_con_SpecialDay"], 4), num(r["AP_sin_SpecialDay"], 4),
            num(r["delta_AP"], 4),
        ])
    tabla(doc, ["Modelo", "AP con SpecialDay", "AP sin SpecialDay", "Diferencia"],
          filas_sd,
          "Tabla VIII. Ablación al retirar SpecialDay con hiperparámetros fijos.", size=7)
    figura(doc, "fig10_ablacion_specialday.png",
           "Fig. 10. Cambio en el AP de validación al retirar SpecialDay.", ancho=3.0)
    par(doc,
        f"En cuanto a SpecialDay, se verificó que sus valores son equiespaciados y que la "
        f"correlación entre la representación numérica estandarizada y la ordinal es "
        f"{num(corr, 6)}; ambas representaciones son, por tanto, idénticas tras el "
        "escalado. La Tabla VIII confirma que retirar SpecialDay produce cambios "
        "despreciables en el AP, por lo que su contribución al modelo es marginal en este "
        "conjunto. La Fig. 2 permite contextualizar este resultado: la cobertura mensual "
        "es desigual y noviembre concentra la mayor tasa de compra, por lo que parte "
        "de la estacionalidad ya queda recogida por el mes y SpecialDay aporta poca "
        "información adicional.")

    # VI. Discusión y conclusiones
    titulo_seccion(doc, "Discusión y conclusiones", "VI")
    par(doc,
        "El diseño experimental permitió comparar cinco familias de modelos bajo un "
        "protocolo común, con el preprocesamiento y la selección de transformaciones "
        "ajustados exclusivamente dentro de los folds y con un conjunto de prueba aislado. "
        "Random Forest fue el modelo con mejor desempeño, con el mayor AP y ROC-AUC tanto "
        "en validación como en prueba, aunque a costa de un marcado sobreajuste y de un "
        "recall de la clase positiva moderado con el umbral nativo. La red neuronal y SVM "
        "ofrecen un compromiso distinto, con mayor recuperación de la clase minoritaria y "
        "precisión algo menor.")
    par(doc,
        "El análisis de sensibilidad mostró que la señal predictiva se concentra en "
        "PageValues. Esto tiene una consecuencia práctica importante: si el objetivo fuera "
        "decidir durante la sesión, antes de que el usuario complete la compra, esa "
        "variable no estaría disponible y el desempeño caería de forma sustancial. Por "
        "tanto, los resultados aquí presentados corresponden a la clasificación de sesiones "
        "con datos agregados y no deben interpretarse como predicción en tiempo real. Como "
        "trabajo futuro se propone evaluar el desempeño restringiendo los predictores a "
        "información disponible antes del cierre de la sesión, así como calibrar las "
        "probabilidades de SVM y afinar el umbral de decisión según el costo de los errores. "
        "Finalmente, la evaluación de técnicas de reducción de dimensión, prevista para la "
        "siguiente fase del proyecto, permitirá establecer si es posible simplificar el "
        "modelo sin perder desempeño.")

    # Referencias
    titulo_seccion(doc, "Referencias")
    refs = [
        "[1] C. Sakar y Y. Kastro, “Online Shoppers Purchasing Intention Dataset,” UCI "
        "Machine Learning Repository, 2018. doi: 10.24432/C5F88Q.",
        "[2] C. O. Sakar, S. O. Polat, M. Katircioglu y Y. Kastro, “Real-time prediction of "
        "online shoppers’ purchasing intention using multilayer perceptron and LSTM "
        "recurrent neural networks,” Neural Computing and Applications, vol. 31, no. 10, "
        "pp. 6893–6908, 2019. doi: 10.1007/s00521-018-3523-0.",
        "[3] S.-C. Necula, “Exploring the impact of time spent reading product information "
        "on e-commerce websites: A machine learning approach to analyze consumer behavior,” "
        "Behavioral Sciences, vol. 13, no. 6, art. 439, 2023. doi: 10.3390/bs13060439.",
        "[4] M. S. Satu y S. F. Islam, “Modeling online customer purchase intention behavior "
        "applying different feature engineering and classification techniques,” Discover "
        "Artificial Intelligence, vol. 3, art. 36, 2023. doi: 10.1007/s44163-023-00086-0.",
        "[5] J. Lin, “Application of machine learning in predicting consumer behavior and "
        "precision marketing,” PLOS ONE, vol. 20, no. 5, art. e0321854, 2025. doi: "
        "10.1371/journal.pone.0321854.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.first_line_indent = Inches(-0.15)
        run = p.add_run(r)
        run.font.size = Pt(8)

    doc.save(SALIDA_DOCX)
    print("DOCX guardado:", SALIDA_DOCX)
    return SALIDA_DOCX


def convertir_pdf(docx_path: Path, pdf_path: Path):
    import win32com.client

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(str(docx_path.resolve()))
    doc.SaveAs(str(pdf_path.resolve()), FileFormat=17)
    doc.Close()
    word.Quit()
    print("PDF guardado:", pdf_path)


if __name__ == "__main__":
    p = main()
    try:
        convertir_pdf(p, SALIDA_PDF)
    except Exception as e:  # noqa: BLE001
        print("No se pudo convertir a PDF:", e)

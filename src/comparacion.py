"""Modelo de comparación: LightGBM (gradient boosting) sobre los datos limpios."""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import COLUMNA_OBJETIVO, SEMILLA
from src.limpieza import obtener_datos_limpios
from src.woe import VARIABLES_CANDIDATAS

# Parámetros fijos: aprendizaje lento y muestreo de filas y columnas para
# reducir el sobreajuste; semilla y modo determinista para reproducibilidad.
PARAMETROS_BASE = {
    "objective": "binary",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEMILLA,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 2,
    "verbose": -1,
}

# Rejilla pequeña: complejidad de cada árbol y tamaño mínimo de sus hojas
REJILLA = [
    {"num_leaves": 15, "min_child_samples": 200},
    {"num_leaves": 31, "min_child_samples": 200},
    {"num_leaves": 15, "min_child_samples": 1000},
    {"num_leaves": 31, "min_child_samples": 1000},
]


def buscar_hiperparametros(X: pd.DataFrame, y: pd.Series, rejilla: list[dict] = REJILLA,
                           n_folds: int = 5) -> pd.DataFrame:
    """Validación cruzada estratificada SOLO con entrenamiento.

    Para cada combinación de la rejilla, se entrena en 4 partes y se evalúa
    en la quinta (5 veces). La parada temprana elige el número de árboles:
    se deja de agregar árboles cuando el AUC de validación deja de mejorar.
    """
    filas = []
    for parametros in rejilla:
        resultado = lgb.cv(
            {**PARAMETROS_BASE, **parametros, "metric": "auc"},
            lgb.Dataset(X, label=y),
            num_boost_round=3000,
            nfold=n_folds,
            stratified=True,
            seed=SEMILLA,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        clave = next(k for k in resultado if k.endswith("auc-mean"))
        clave_std = next(k for k in resultado if k.endswith("auc-stdv"))
        filas.append({**parametros,
                      "n_arboles": len(resultado[clave]),
                      "auc_validacion": resultado[clave][-1],
                      "auc_desviacion": resultado[clave_std][-1]})
    return pd.DataFrame(filas).sort_values("auc_validacion", ascending=False, ignore_index=True)


def entrenar_lightgbm(X: pd.DataFrame, y: pd.Series, parametros: dict, n_arboles: int) -> lgb.Booster:
    """Entrena el modelo final con todo el conjunto de entrenamiento."""
    return lgb.train({**PARAMETROS_BASE, **parametros}, lgb.Dataset(X, label=y),
                     num_boost_round=n_arboles)


def importancia_variables(modelo: lgb.Booster) -> pd.DataFrame:
    """Importancia por ganancia: cuánto mejora el modelo cada variable, en porcentaje."""
    ganancia = pd.Series(modelo.feature_importance(importance_type="gain"),
                         index=modelo.feature_name())
    tabla = (ganancia / ganancia.sum() * 100).sort_values(ascending=False).to_frame("importancia_pct")
    tabla.index.name = "variable"
    return tabla


def graficar_importancia(tabla: pd.DataFrame):
    """Barras horizontales de importancia por ganancia."""
    datos = tabla.sort_values("importancia_pct")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(datos.index, datos["importancia_pct"], color="#2a78d6", height=0.6)
    for posicion, valor in enumerate(datos["importancia_pct"]):
        ax.annotate(f"{valor:.1f} %", xy=(valor, posicion), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Importancia (% de la ganancia total)")
    ax.set_title("LightGBM: importancia de cada variable", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def contribuciones_cliente(modelo: lgb.Booster, X: pd.DataFrame, id_cliente) -> pd.DataFrame:
    """Contribución de cada variable a la predicción de un cliente (valores SHAP).

    Están en escala de logaritmo de momios de ser malo: positivo = la variable
    sube el riesgo del cliente; negativo = lo baja.
    """
    fila = X.loc[[id_cliente]]
    valores = modelo.predict(fila, pred_contrib=True)[0]
    tabla = pd.DataFrame({"valor": fila.iloc[0].to_numpy(), "contribucion": valores[:-1]},
                         index=X.columns)
    tabla.index.name = "variable"
    return tabla.sort_values("contribucion", ascending=False)


def construir_modelo_comparacion():
    """Reconstruye el modelo de comparación de forma determinista.

    Devuelve un diccionario con el modelo, la búsqueda de hiperparámetros y
    la PD estimada para entrenamiento y prueba.
    """
    train, test = obtener_datos_limpios()
    X_train, y_train = train[VARIABLES_CANDIDATAS], train[COLUMNA_OBJETIVO]
    X_test = test[VARIABLES_CANDIDATAS]

    busqueda = buscar_hiperparametros(X_train, y_train)
    mejor = busqueda.iloc[0]
    parametros = {"num_leaves": int(mejor["num_leaves"]),
                  "min_child_samples": int(mejor["min_child_samples"])}
    modelo = entrenar_lightgbm(X_train, y_train, parametros, int(mejor["n_arboles"]))
    return {
        "modelo": modelo,
        "busqueda": busqueda,
        "parametros": parametros,
        "pd_train": pd.Series(modelo.predict(X_train), index=train.index, name="pd_lightgbm"),
        "pd_test": pd.Series(modelo.predict(X_test), index=test.index, name="pd_lightgbm"),
    }

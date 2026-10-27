"""Métricas de la industria para modelos de riesgo: AUC, Gini, KS y PSI.

Convención: `riesgo` es cualquier valor donde MAYOR = MÁS RIESGO (por ejemplo,
la PD). Para el scorecard se usa el puntaje con signo negativo, porque ahí
más puntos = menos riesgo.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import SEMILLA

COLORES = ["#2a78d6", "#eb6834"]


def ks(y: pd.Series, riesgo: pd.Series) -> float:
    """KS: máxima distancia entre el % acumulado de malos y el % acumulado de buenos."""
    tasa_buenos, tasa_malos, _ = roc_curve(y, riesgo)
    return float(np.max(tasa_malos - tasa_buenos))


def metricas(y: pd.Series, riesgo: pd.Series) -> dict:
    """AUC, Gini (= 2·AUC − 1) y KS."""
    auc = roc_auc_score(y, riesgo)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks(y, riesgo)}


def tabla_metricas(evaluaciones: dict) -> pd.DataFrame:
    """Métricas para varios (modelo, muestra): {(modelo, muestra): (y, riesgo)}."""
    filas = [{"modelo": modelo, "muestra": muestra, **metricas(y, riesgo)}
             for (modelo, muestra), (y, riesgo) in evaluaciones.items()]
    return pd.DataFrame(filas).set_index(["modelo", "muestra"])


def bootstrap_diferencia_auc(y: pd.Series, riesgo_a: pd.Series, riesgo_b: pd.Series,
                             n_repeticiones: int = 300, semilla: int = SEMILLA) -> pd.Series:
    """Intervalo de confianza de 95 % para AUC(b) − AUC(a), remuestreando clientes."""
    generador = np.random.default_rng(semilla)
    y, a, b = y.to_numpy(), riesgo_a.to_numpy(), riesgo_b.to_numpy()
    diferencias = []
    for _ in range(n_repeticiones):
        indices = generador.integers(0, len(y), len(y))
        diferencias.append(roc_auc_score(y[indices], b[indices]) - roc_auc_score(y[indices], a[indices]))
    diferencias = np.array(diferencias)
    return pd.Series({
        "diferencia_auc": roc_auc_score(y, b) - roc_auc_score(y, a),
        "limite_inferior_95": np.percentile(diferencias, 2.5),
        "limite_superior_95": np.percentile(diferencias, 97.5),
    })


def tabla_deciles(y: pd.Series, riesgo: pd.Series, n_grupos: int = 10) -> pd.DataFrame:
    """Tabla de ganancias: clientes ordenados de mayor a menor riesgo, en deciles.

    Muestra qué porcentaje de todos los malos se concentra en los clientes
    más riesgosos (captura acumulada) y el KS en cada corte.
    """
    rango = riesgo.rank(method="first", ascending=False)
    decil = pd.qcut(rango, q=n_grupos, labels=range(1, n_grupos + 1))
    tabla = pd.DataFrame({"decil": decil, "malo": y}).groupby("decil", observed=True)["malo"].agg(
        clientes="size", malos="sum")
    tabla["buenos"] = tabla["clientes"] - tabla["malos"]
    tabla["tasa_malos"] = tabla["malos"] / tabla["clientes"]
    tabla["pct_malos_acum"] = tabla["malos"].cumsum() / tabla["malos"].sum()
    tabla["pct_buenos_acum"] = tabla["buenos"].cumsum() / tabla["buenos"].sum()
    tabla["ks"] = tabla["pct_malos_acum"] - tabla["pct_buenos_acum"]
    tabla.index.name = "decil (1 = más riesgo)"
    return tabla


def psi(esperado: pd.Series, actual: pd.Series) -> float:
    """PSI entre dos distribuciones de proporciones por grupo.

    PSI = suma de (actual − esperado) × ln(actual / esperado).
    Referencia de la industria: < 0.10 estable; 0.10–0.25 revisar; > 0.25 inestable.
    """
    esperado, actual = esperado.align(actual, fill_value=0)
    esperado = esperado.clip(lower=1e-6)
    actual = actual.clip(lower=1e-6)
    return float(((actual - esperado) * np.log(actual / esperado)).sum())


def estabilidad(psi_valor: float) -> str:
    if psi_valor < 0.10:
        return "Estable"
    if psi_valor < 0.25:
        return "Revisar"
    return "Inestable"


def psi_puntaje(puntaje_base: pd.Series, puntaje_nuevo: pd.Series, n_grupos: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI del puntaje, con rangos definidos por los deciles de la muestra base."""
    cortes = np.unique(np.quantile(puntaje_base, np.linspace(0, 1, n_grupos + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    base = pd.cut(puntaje_base, cortes).value_counts(normalize=True, sort=False)
    nuevo = pd.cut(puntaje_nuevo, cortes).value_counts(normalize=True, sort=False)
    tabla = pd.DataFrame({"pct_entrenamiento": base, "pct_prueba": nuevo})
    tabla.index = tabla.index.astype(str)
    tabla.index.name = "rango_puntaje"
    return psi(base, nuevo), tabla


def psi_variables(agrupador, datos_base: pd.DataFrame, datos_nuevos: pd.DataFrame,
                  variables: list[str]) -> pd.DataFrame:
    """PSI de cada variable, usando sus tramos del scorecard."""
    filas = []
    for var in variables:
        base = agrupador.asignar_tramos(datos_base[var], var).astype(str).value_counts(normalize=True)
        nuevo = agrupador.asignar_tramos(datos_nuevos[var], var).astype(str).value_counts(normalize=True)
        valor = psi(base, nuevo)
        filas.append({"variable": var, "psi": valor, "estabilidad": estabilidad(valor)})
    return pd.DataFrame(filas).set_index("variable").sort_values("psi", ascending=False)


def graficar_roc(curvas: dict):
    """Curvas ROC: {nombre: (y, riesgo)}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (nombre, (y, riesgo)), color in zip(curvas.items(), COLORES):
        tasa_buenos, tasa_malos, _ = roc_curve(y, riesgo)
        ax.plot(tasa_buenos, tasa_malos, color=color, linewidth=2,
                label=f"{nombre} (AUC = {roc_auc_score(y, riesgo):.4f})")
    ax.plot([0, 1], [0, 1], color="#52514e", linestyle="--", linewidth=1, label="Al azar (AUC = 0.5)")
    ax.set_xlabel("% de buenos rechazados (falsos positivos)")
    ax.set_ylabel("% de malos detectados (verdaderos positivos)")
    ax.set_title("Curva ROC en prueba", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graficar_ks(y: pd.Series, puntaje: pd.Series, titulo: str = "KS del scorecard en prueba"):
    """% acumulado de malos y de buenos por puntaje; el KS es la mayor distancia."""
    orden = np.sort(puntaje.unique())
    malos = puntaje[y == 1]
    buenos = puntaje[y == 0]
    acum_malos = np.searchsorted(np.sort(malos), orden, side="right") / len(malos)
    acum_buenos = np.searchsorted(np.sort(buenos), orden, side="right") / len(buenos)
    distancia = acum_malos - acum_buenos
    i = int(np.argmax(distancia))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(orden, acum_malos * 100, color=COLORES[1], linewidth=2, label="Malos (% acumulado)")
    ax.plot(orden, acum_buenos * 100, color=COLORES[0], linewidth=2, label="Buenos (% acumulado)")
    ax.vlines(orden[i], acum_buenos[i] * 100, acum_malos[i] * 100, color="#52514e", linestyle="--")
    ax.annotate(f"KS = {distancia[i]:.4f}\nen {orden[i]:.0f} puntos", xy=(orden[i], (acum_malos[i] + acum_buenos[i]) * 50),
                xytext=(-8, 0), textcoords="offset points", fontsize=9, color="#52514e",
                va="center", ha="right")
    ax.set_xlabel("Puntaje")
    ax.set_ylabel("% acumulado")
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

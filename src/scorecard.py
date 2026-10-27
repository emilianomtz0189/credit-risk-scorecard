"""Regresión logística sobre variables WoE y conversión a scorecard de puntos.

Escala del scorecard (estándar de la industria, parametrizable):
    - PUNTAJE_BASE puntos equivalen a momios de MOMIOS_BASE buenos por cada malo.
    - Cada PDO puntos (Points to Double the Odds) los momios se duplican.

    puntaje = offset + factor * ln(momios de bueno)
    factor  = PDO / ln(2)
    offset  = PUNTAJE_BASE - factor * ln(MOMIOS_BASE)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import COLUMNA_OBJETIVO
from src.limpieza import obtener_datos_limpios
from src.woe import ETIQUETA_FALTANTE, preparar_woe

PUNTAJE_BASE = 600
MOMIOS_BASE = 50   # 50 buenos por cada malo en el puntaje base
PDO = 20           # Cada 20 puntos, los momios se duplican

FACTOR = PDO / np.log(2)
OFFSET = PUNTAJE_BASE - FACTOR * np.log(MOMIOS_BASE)

P_VALOR_MAXIMO = 0.05


class RegresionLogistica:
    """Regresión logística por máxima verosimilitud (método de Newton-Raphson).

    Da los mismos coeficientes, errores estándar y p-valores que
    statsmodels.Logit, sin penalización (a diferencia de scikit-learn, que por
    omisión regulariza los coeficientes).
    """

    def __init__(self, max_iter: int = 50, tolerancia: float = 1e-8):
        self.max_iter = max_iter
        self.tolerancia = tolerancia

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegresionLogistica":
        self.variables_ = list(X.columns)
        matriz = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
        objetivo = y.to_numpy(dtype=float)
        beta = np.zeros(matriz.shape[1])

        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-matriz @ beta))
            gradiente = matriz.T @ (objetivo - p)
            hessiana = (matriz * (p * (1 - p))[:, None]).T @ matriz
            paso = np.linalg.solve(hessiana, gradiente)
            beta += paso
            if np.max(np.abs(paso)) < self.tolerancia:
                break

        p = 1 / (1 + np.exp(-matriz @ beta))
        hessiana = (matriz * (p * (1 - p))[:, None]).T @ matriz
        errores = np.sqrt(np.diag(np.linalg.inv(hessiana)))

        nombres = ["intercepto"] + self.variables_
        self.coeficientes_ = pd.Series(beta, index=nombres)
        self.errores_ = pd.Series(errores, index=nombres)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """PD estimada: probabilidad de ser malo."""
        z = self.coeficientes_["intercepto"] + X[self.variables_].to_numpy() @ self.coeficientes_[self.variables_].to_numpy()
        return 1 / (1 + np.exp(-z))

    def tabla_coeficientes(self) -> pd.DataFrame:
        """Coeficiente, error estándar, estadístico z (Wald) y p-valor."""
        z = self.coeficientes_ / self.errores_
        return pd.DataFrame({
            "coeficiente": self.coeficientes_,
            "error_estandar": self.errores_,
            "z": z,
            "p_valor": 2 * (1 - stats.norm.cdf(np.abs(z))),
        })


def ajustar_logistica(datos_woe: pd.DataFrame, variables: list[str],
                      p_valor_maximo: float = P_VALOR_MAXIMO):
    """Ajusta la regresión y elimina, de una en una, variables problemáticas.

    Con la convención WoE = ln(% buenos / % malos), un WoE mayor significa
    menos riesgo, así que todos los coeficientes deben ser NEGATIVOS.
    Se elimina primero cualquier variable con signo positivo (síntoma de
    multicolinealidad) y después la de mayor p-valor si supera el máximo.
    """
    variables = list(variables)
    eliminadas = []
    while True:
        modelo = RegresionLogistica().fit(datos_woe[variables], datos_woe[COLUMNA_OBJETIVO])
        tabla = modelo.tabla_coeficientes().drop(index="intercepto")

        signo_incorrecto = tabla[tabla["coeficiente"] > 0]
        if not signo_incorrecto.empty:
            peor = signo_incorrecto["p_valor"].idxmax()
            eliminadas.append((peor, "coeficiente con signo positivo"))
        elif (tabla["p_valor"] > p_valor_maximo).any():
            peor = tabla["p_valor"].idxmax()
            eliminadas.append((peor, f"p-valor {tabla.loc[peor, 'p_valor']:.3f}"))
        else:
            break
        variables.remove(peor)

    tabla_eliminadas = pd.DataFrame(eliminadas, columns=["variable", "motivo"])
    return modelo, variables, tabla_eliminadas


def tabla_puntos(modelo: RegresionLogistica, agrupador) -> pd.DataFrame:
    """Scorecard: puntos de cada tramo de cada variable.

    puntos = (-(coeficiente * WoE) - intercepto / n) * factor + offset / n
    La suma de los puntos de un cliente es su puntaje total.
    """
    b0 = modelo.coeficientes_["intercepto"]
    n = len(modelo.variables_)
    filas = []
    for var in modelo.variables_:
        b = modelo.coeficientes_[var]
        tabla = agrupador.tablas_[var]
        woes = tabla["woe"].to_dict()
        if ETIQUETA_FALTANTE not in woes:
            woes[ETIQUETA_FALTANTE] = 0.0   # Sin faltantes en entrenamiento: riesgo promedio
        for tramo, woe in woes.items():
            puntos = (-(b * woe) - b0 / n) * FACTOR + OFFSET / n
            clientes = tabla["clientes"].get(tramo, 0)
            tasa = tabla["tasa_malos"].get(tramo, np.nan)
            filas.append({"variable": var, "tramo": tramo, "clientes": clientes,
                          "tasa_malos": tasa, "woe": woe, "puntos": int(round(puntos))})
    return pd.DataFrame(filas).set_index(["variable", "tramo"])


def puntaje_clientes(datos_limpios: pd.DataFrame, agrupador, puntos: pd.DataFrame) -> pd.DataFrame:
    """Puntos por variable y puntaje total de cada cliente, según el scorecard."""
    resultado = pd.DataFrame(index=datos_limpios.index)
    for var in puntos.index.get_level_values("variable").unique():
        puntos_var = puntos.loc[var, "puntos"]
        tramos = agrupador.asignar_tramos(datos_limpios[var], var).astype(str)
        resultado[var] = tramos.map(puntos_var).astype(int)
    resultado["puntaje"] = resultado.sum(axis=1)
    return resultado


def desglose_cliente(id_cliente, datos_limpios: pd.DataFrame, agrupador,
                     puntos: pd.DataFrame) -> pd.DataFrame:
    """Valor, tramo y puntos de cada variable para un cliente.

    Sirve para explicar una decisión: las variables con menos puntos son las
    razones principales de un puntaje bajo (reason codes).
    """
    filas = []
    for var in puntos.index.get_level_values("variable").unique():
        valor = datos_limpios.loc[id_cliente, var]
        tramo = str(agrupador.asignar_tramos(datos_limpios.loc[[id_cliente], var], var).iloc[0])
        filas.append({"variable": var, "valor": valor, "tramo": tramo,
                      "puntos": puntos.loc[(var, tramo), "puntos"]})
    tabla = pd.DataFrame(filas).set_index("variable")
    return tabla


def pd_desde_puntaje(puntaje) -> np.ndarray:
    """PD implícita en un puntaje: invierte la escala del scorecard."""
    return 1 / (1 + np.exp((np.asarray(puntaje) - OFFSET) / FACTOR))


def tabla_calibracion(puntaje: pd.Series, y: pd.Series, n_grupos: int = 10) -> pd.DataFrame:
    """Compara la PD del scorecard con la tasa de malos real, por rangos de puntaje."""
    rangos = pd.qcut(puntaje, q=n_grupos, duplicates="drop")
    datos = pd.DataFrame({"rango": rangos, "malo": y, "pd": pd_desde_puntaje(puntaje)})
    tabla = datos.groupby("rango", observed=True).agg(
        clientes=("malo", "size"), malos=("malo", "sum"),
        tasa_malos_real=("malo", "mean"), pd_promedio=("pd", "mean"))
    tabla.index = [f"{int(np.ceil(i.left))}–{int(i.right)}" for i in tabla.index]
    tabla.index.name = "rango_puntaje"
    return tabla


def graficar_distribucion_puntajes(puntaje: pd.Series, y: pd.Series):
    """Histograma de puntajes de clientes buenos y malos (densidad)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    rango = (puntaje.min(), puntaje.max())
    ax.hist(puntaje[y == 0], bins=40, range=rango, density=True, alpha=0.6,
            color="#2a78d6", label="Buenos")
    ax.hist(puntaje[y == 1], bins=40, range=rango, density=True, alpha=0.6,
            color="#eb6834", label="Malos")
    ax.set_xlabel("Puntaje del scorecard")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de puntajes: buenos vs. malos", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def construir_scorecard():
    """Reconstruye todo el scorecard de forma determinista (para notebooks posteriores).

    Devuelve un diccionario con el agrupador, el modelo, la tabla de puntos y
    los puntajes de entrenamiento y prueba.
    """
    agrupador, seleccion, train_woe, test_woe = preparar_woe()
    train, test = obtener_datos_limpios()
    modelo, variables, eliminadas = ajustar_logistica(train_woe, seleccion)
    puntos = tabla_puntos(modelo, agrupador)
    return {
        "agrupador": agrupador,
        "modelo": modelo,
        "variables": variables,
        "eliminadas": eliminadas,
        "puntos": puntos,
        "train_woe": train_woe,
        "test_woe": test_woe,
        "puntaje_train": puntaje_clientes(train, agrupador, puntos),
        "puntaje_test": puntaje_clientes(test, agrupador, puntos),
    }

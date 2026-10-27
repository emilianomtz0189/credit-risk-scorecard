"""Pérdida esperada (PD × LGD × EAD) y propuesta de puntos de corte.

El dataset no trae montos de crédito ni recuperaciones, así que LGD, EAD y
margen son SUPUESTOS explícitos y parametrizables. Los resultados se
presentan con análisis de sensibilidad para que se vea cuánto dependen de ellos.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Supuestos de negocio (horizonte de 2 años, igual que la definición de malo) ---
EAD_POR_CLIENTE = 50_000   # Monto expuesto por cliente, en pesos (crédito hipotético)
LGD = 0.45                 # Pérdida en caso de incumplimiento (referencia: Basilea, deuda no garantizada)
MARGEN = 0.15              # Ingreso neto de un cliente que paga, como % de la EAD, en el horizonte

# Bandas de puntaje de 20 puntos, con los extremos agrupados
CORTES_BANDAS = [-np.inf, 500, 520, 540, 560, 580, 600, 620, np.inf]


def _etiquetas_bandas(cortes: list[float]) -> list[str]:
    etiquetas = []
    for izq, der in zip(cortes[:-1], cortes[1:]):
        if np.isinf(izq):
            etiquetas.append(f"< {int(der)}")
        elif np.isinf(der):
            etiquetas.append(f"{int(izq)}+")
        else:
            etiquetas.append(f"{int(izq)}–{int(der) - 1}")
    return etiquetas


def asignar_banda(puntaje: pd.Series, cortes: list[float] = CORTES_BANDAS) -> pd.Series:
    """Banda de puntaje de cada cliente (intervalos [a, b))."""
    return pd.cut(puntaje, bins=cortes, right=False, labels=_etiquetas_bandas(cortes))


def calibrar_pd_por_banda(puntaje: pd.Series, y: pd.Series) -> pd.DataFrame:
    """PD calibrada = tasa de malos observada en cada banda (con entrenamiento)."""
    tabla = (pd.DataFrame({"banda": asignar_banda(puntaje), "malo": y})
             .groupby("banda", observed=False)["malo"]
             .agg(clientes="size", malos="sum"))
    tabla["pd_calibrada"] = tabla["malos"] / tabla["clientes"]
    return tabla


def pd_equilibrio(margen: float = MARGEN, lgd: float = LGD) -> float:
    """PD a partir de la cual un cliente deja de ser rentable.

    Utilidad esperada = EAD × [(1 − PD) × margen − PD × LGD] = 0
    ⇒ PD* = margen / (margen + LGD)
    """
    return margen / (margen + lgd)


def tabla_perdida_esperada(puntaje: pd.Series, pd_banda: pd.Series, ead: float = EAD_POR_CLIENTE,
                           lgd: float = LGD, margen: float = MARGEN) -> pd.DataFrame:
    """Pérdida esperada y utilidad esperada por banda, de la mejor a la peor.

    Las columnas acumuladas responden: "si apruebo desde la mejor banda hasta
    esta, ¿qué % de clientes apruebo y cuánta pérdida y utilidad espero?".
    """
    clientes = asignar_banda(puntaje).value_counts().reindex(pd_banda.index).fillna(0)
    tabla = pd.DataFrame({"clientes": clientes.astype(int), "pd": pd_banda})
    tabla["exposicion"] = tabla["clientes"] * ead
    tabla["perdida_esperada"] = tabla["pd"] * lgd * tabla["exposicion"]
    tabla["ingreso_esperado"] = (1 - tabla["pd"]) * margen * tabla["exposicion"]
    tabla["utilidad_esperada"] = tabla["ingreso_esperado"] - tabla["perdida_esperada"]

    tabla = tabla.iloc[::-1]   # De la mejor banda (más puntos) a la peor
    tabla["pct_aprobacion_acum"] = tabla["clientes"].cumsum() / tabla["clientes"].sum()
    tabla["perdida_esperada_acum"] = tabla["perdida_esperada"].cumsum()
    tabla["utilidad_esperada_acum"] = tabla["utilidad_esperada"].cumsum()
    return tabla


def corte_por_utilidad(pd_banda: pd.Series, margen: float = MARGEN, lgd: float = LGD,
                       cortes: list[float] = CORTES_BANDAS) -> float:
    """Puntaje mínimo de aprobación: inicio de la banda más baja con utilidad esperada positiva.

    Se recorre de la mejor banda a la peor y se detiene en la primera que no es rentable.
    """
    limite = pd_equilibrio(margen, lgd)
    inicios = cortes[:-1]
    corte = np.inf   # Si ninguna banda es rentable, no se aprueba a nadie
    for inicio, pd_valor in reversed(list(zip(inicios, pd_banda.to_numpy()))):
        if pd_valor >= limite:
            break
        corte = inicio
    return corte


def puntaje_ks(puntaje: pd.Series, y: pd.Series) -> float:
    """Puntaje donde se alcanza el KS: máxima distancia entre malos y buenos acumulados."""
    orden = np.sort(puntaje.unique())
    malos, buenos = np.sort(puntaje[y == 1]), np.sort(puntaje[y == 0])
    acum_malos = np.searchsorted(malos, orden, side="right") / len(malos)
    acum_buenos = np.searchsorted(buenos, orden, side="right") / len(buenos)
    return float(orden[np.argmax(acum_malos - acum_buenos)])


def asignar_zona(puntaje: pd.Series, corte_rechazo: float, corte_aprobacion: float) -> pd.Series:
    """Rechazo (< corte_rechazo), Revisión manual (entre ambos), Aprobación automática (≥ corte_aprobacion)."""
    zona = np.where(puntaje < corte_rechazo, "Rechazo",
                    np.where(puntaje < corte_aprobacion, "Revisión manual", "Aprobación automática"))
    orden = ["Aprobación automática", "Revisión manual", "Rechazo"]
    return pd.Series(pd.Categorical(zona, categories=orden, ordered=True), index=puntaje.index)


def evaluar_zonas(puntaje: pd.Series, y: pd.Series, corte_rechazo: float, corte_aprobacion: float,
                  ead: float = EAD_POR_CLIENTE, lgd: float = LGD, margen: float = MARGEN) -> pd.DataFrame:
    """Resultado REAL por zona: malos observados, pérdida, ingreso y utilidad."""
    datos = pd.DataFrame({"zona": asignar_zona(puntaje, corte_rechazo, corte_aprobacion), "malo": y})
    tabla = datos.groupby("zona", observed=False)["malo"].agg(clientes="size", malos="sum")
    tabla["pct_clientes"] = tabla["clientes"] / tabla["clientes"].sum()
    tabla["tasa_malos"] = tabla["malos"] / tabla["clientes"]
    tabla["pct_malos_total"] = tabla["malos"] / tabla["malos"].sum()
    tabla["perdida"] = tabla["malos"] * lgd * ead
    tabla["ingreso"] = (tabla["clientes"] - tabla["malos"]) * margen * ead
    tabla["utilidad"] = tabla["ingreso"] - tabla["perdida"]
    return tabla


def comparar_escenarios(zonas: pd.DataFrame) -> pd.DataFrame:
    """Utilidad real si se aprueba a todos, solo la zona automática, o automática + revisión."""
    escenarios = {
        "Aprobar a todos": ["Aprobación automática", "Revisión manual", "Rechazo"],
        "Aprobación automática + revisión": ["Aprobación automática", "Revisión manual"],
        "Solo aprobación automática": ["Aprobación automática"],
    }
    filas = []
    for nombre, incluidas in escenarios.items():
        parte = zonas.loc[incluidas]
        clientes, malos = parte["clientes"].sum(), parte["malos"].sum()
        filas.append({"escenario": nombre,
                      "pct_aprobados": clientes / zonas["clientes"].sum(),
                      "tasa_malos_aprobados": malos / clientes,
                      "perdida": parte["perdida"].sum(),
                      "utilidad": parte["utilidad"].sum()})
    return pd.DataFrame(filas).set_index("escenario")


def sensibilidad(pd_banda: pd.Series, puntaje: pd.Series, lgds: list[float], margenes: list[float]) -> pd.DataFrame:
    """Punto de corte y % de aprobación para distintos supuestos de LGD y margen."""
    filas = []
    for lgd in lgds:
        for margen in margenes:
            corte = corte_por_utilidad(pd_banda, margen, lgd)
            filas.append({"lgd": lgd, "margen": margen,
                          "pd_equilibrio": pd_equilibrio(margen, lgd),
                          "corte_rechazo": corte,
                          "pct_aprobacion": (puntaje >= corte).mean()})
    return pd.DataFrame(filas)


def graficar_perdida_por_banda(tabla: pd.DataFrame):
    """Pérdida esperada (millones de pesos) y PD por banda de puntaje."""
    datos = tabla.iloc[::-1]   # De la peor a la mejor banda, de izquierda a derecha
    fig, ax = plt.subplots(figsize=(8, 3.6))
    posiciones = np.arange(len(datos))
    ax.bar(posiciones, datos["perdida_esperada"] / 1e6, color="#eb6834", width=0.7)
    for x, (perdida, pd_valor) in enumerate(zip(datos["perdida_esperada"], datos["pd"])):
        ax.annotate(f"PD {pd_valor:.1%}", xy=(x, perdida / 1e6), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(posiciones)
    ax.set_xticklabels(datos.index.astype(str), fontsize=9)
    ax.set_xlabel("Banda de puntaje")
    ax.set_ylabel("Pérdida esperada (millones $)")
    ax.set_title("Pérdida esperada por banda de puntaje", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

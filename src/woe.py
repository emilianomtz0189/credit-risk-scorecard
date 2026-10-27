"""Agrupación de variables (binning), WoE e IV para el scorecard.

Convención usada (la del libro de Siddiqi, estándar en la industria):
    WoE = ln(% de buenos del tramo / % de malos del tramo)
    IV  = suma de (% buenos - % malos) * WoE

WoE positivo = tramo con menos riesgo que el promedio.
WoE negativo = tramo con más riesgo que el promedio.
WoE cero     = riesgo igual al promedio.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ARCHIVO_TEST_WOE,
    ARCHIVO_TRAIN_WOE,
    COLUMNA_OBJETIVO,
)
from src.limpieza import obtener_datos_limpios

ETIQUETA_FALTANTE = "Faltante"

# Variables candidatas al modelo
VARIABLES_CANDIDATAS = [
    "uso_credito_revolvente", "edad", "atrasos_30_59", "razon_deuda",
    "ingreso_mensual", "num_creditos_abiertos", "atrasos_90",
    "num_creditos_inmobiliarios", "atrasos_60_89", "num_dependientes",
]
# Variables con relación en U justificada por negocio: no se fuerza tendencia
# monótona (0 créditos inmobiliarios = sin patrimonio; 4 o más = sobreendeudamiento)
NO_MONOTONICAS = ["num_creditos_inmobiliarios"]

IV_MINIMO = 0.02        # Debajo de este valor la variable no es predictiva
CORRELACION_MAXIMA = 0.7  # Entre variables WoE; arriba se descarta la de menor IV


def poder_predictivo(iv: float) -> str:
    """Clasificación del IV usada en la industria (Siddiqi)."""
    if iv < 0.02:
        return "Sin poder predictivo"
    if iv < 0.1:
        return "Débil"
    if iv < 0.3:
        return "Medio"
    if iv < 0.5:
        return "Fuerte"
    return "Muy fuerte (revisar)"


def tabla_woe(grupos: pd.Series, y: pd.Series, orden: list[str]) -> pd.DataFrame:
    """Tabla con clientes, malos, tasa de malos, WoE e IV por grupo."""
    datos = pd.DataFrame({"grupo": grupos.astype(str), "malo": y})
    tabla = datos.groupby("grupo")["malo"].agg(clientes="size", malos="sum")
    tabla = tabla.reindex([g for g in orden if g in tabla.index])

    tabla["buenos"] = tabla["clientes"] - tabla["malos"]
    tabla["pct_clientes"] = tabla["clientes"] / tabla["clientes"].sum()
    tabla["tasa_malos"] = tabla["malos"] / tabla["clientes"]

    # Si un grupo no tiene buenos o malos, el WoE sería infinito: se usa 0.5
    buenos = tabla["buenos"].where(tabla["buenos"] > 0, 0.5)
    malos = tabla["malos"].where(tabla["malos"] > 0, 0.5)
    tabla["dist_buenos"] = buenos / tabla["buenos"].sum()
    tabla["dist_malos"] = malos / tabla["malos"].sum()

    tabla["woe"] = np.log(tabla["dist_buenos"] / tabla["dist_malos"])
    tabla["iv"] = (tabla["dist_buenos"] - tabla["dist_malos"]) * tabla["woe"]
    tabla.index.name = "tramo"
    return tabla


def _formato(valor: float) -> str:
    return f"{valor:,.0f}" if abs(valor) >= 100 else f"{valor:.2f}"


def _etiquetas(cortes: list[float], entera: bool) -> list[str]:
    """Etiquetas legibles para tramos (a, b] definidos por los cortes."""
    etiquetas = []
    for izq, der in zip(cortes[:-1], cortes[1:]):
        if entera:
            inicio = None if np.isinf(izq) else int(izq) + 1
            fin = None if np.isinf(der) else int(der)
            if fin is None:
                etiquetas.append(f"{inicio:,}+" if inicio is not None else "Todos")
            elif inicio is None:
                etiquetas.append(f"≤ {fin:,}" if fin > 0 else f"{fin}")
            elif inicio == fin:
                etiquetas.append(f"{inicio:,}")
            else:
                etiquetas.append(f"{inicio:,}–{fin:,}")
        else:
            if np.isinf(izq) and np.isinf(der):
                etiquetas.append("Todos")
            elif np.isinf(izq):
                etiquetas.append(f"≤ {_formato(der)}")
            elif np.isinf(der):
                etiquetas.append(f"> {_formato(izq)}")
            else:
                etiquetas.append(f"{_formato(izq)} – {_formato(der)}")
    return etiquetas


class AgrupadorWoE:
    """Agrupa cada variable en tramos y calcula su WoE e IV.

    Pasos por variable (todo con la muestra de entrenamiento):
    1. Tramos finos (fine classing): hasta `n_tramos_iniciales` tramos por
       cuantiles, o un tramo por valor si la variable tiene pocos valores.
    2. Tramos gruesos (coarse classing): se fusionan tramos vecinos hasta que
       a) cada tramo tenga al menos `pct_minimo` de los clientes y buenos y malos, y
       b) la tasa de malos tenga una tendencia monótona (siempre sube o siempre baja).
       c) haya como máximo `max_tramos` tramos (se fusionan los vecinos de
          tasa de malos más parecida), para un scorecard simple y estable.
    3. Los faltantes forman su propio tramo. Si son menos de
       `min_clientes_faltante`, no hay evidencia suficiente y reciben WoE = 0.
    """

    def __init__(self, n_tramos_iniciales: int = 20, pct_minimo: float = 0.02,
                 max_tramos: int = 8, min_clientes_faltante: int = 50,
                 no_monotonicas: list[str] | None = None):
        self.n_tramos_iniciales = n_tramos_iniciales
        self.pct_minimo = pct_minimo
        self.max_tramos = max_tramos
        self.min_clientes_faltante = min_clientes_faltante
        self.no_monotonicas = set(no_monotonicas or [])
        self.cortes_ = {}
        self.enteras_ = {}
        self.tablas_ = {}

    # --- Construcción de tramos -------------------------------------------------
    def _cortes_iniciales(self, x: pd.Series) -> list[float]:
        valores = np.sort(x.dropna().unique())
        if len(valores) <= self.n_tramos_iniciales:
            interiores = list(valores[:-1])           # Un tramo por valor
        else:
            cuantiles = np.linspace(0, 1, self.n_tramos_iniciales + 1)[1:-1]
            interiores = list(np.unique(np.quantile(x.dropna(), cuantiles)))
        return [-np.inf] + [float(c) for c in interiores] + [np.inf]

    @staticmethod
    def _estadisticos(x: pd.Series, y: pd.Series, cortes: list[float]) -> pd.DataFrame:
        con_dato = x.notna()
        tramo = pd.cut(x[con_dato], bins=cortes, labels=False, right=True)
        est = y[con_dato].groupby(tramo).agg(clientes="size", malos="sum")
        est = est.reindex(range(len(cortes) - 1), fill_value=0)
        est["buenos"] = est["clientes"] - est["malos"]
        est["tasa"] = est["malos"] / est["clientes"].replace(0, np.nan)
        return est

    def _agrupar(self, x: pd.Series, y: pd.Series, monotonica: bool) -> list[float]:
        cortes = self._cortes_iniciales(x)
        minimo = self.pct_minimo * len(x)
        # Dirección de la tendencia: signo de la correlación de Spearman con el objetivo
        tendencia = np.sign(x.corr(y, method="spearman")) or 1.0

        while len(cortes) > 2:
            est = self._estadisticos(x, y, cortes)

            # a) Tramo más pequeño que no cumple tamaño mínimo o sin buenos/malos
            invalidos = est[(est["clientes"] < minimo) | (est["buenos"] == 0) | (est["malos"] == 0)]
            if not invalidos.empty:
                i = int(invalidos["clientes"].idxmin())
                vecinos = [j for j in (i - 1, i + 1) if 0 <= j < len(est)]
                # Se fusiona con el vecino de tasa de malos más parecida
                j = min(vecinos, key=lambda v: abs(np.nan_to_num(est.loc[v, "tasa"]) -
                                                    np.nan_to_num(est.loc[i, "tasa"])))
                del cortes[max(i, j)]
                continue

            # b) Par de tramos vecinos que rompe la tendencia monótona
            if monotonica:
                diferencias = np.diff(est["tasa"].to_numpy())
                violaciones = np.where(diferencias * tendencia < 0)[0]
                if len(violaciones) > 0:
                    # Se fusiona el par que rompe la tendencia con la menor diferencia
                    k = violaciones[np.argmin(np.abs(diferencias[violaciones]))]
                    del cortes[k + 1]
                    continue

            # c) Demasiados tramos: se fusionan los vecinos más parecidos
            if len(est) > self.max_tramos:
                k = int(np.argmin(np.abs(np.diff(est["tasa"].to_numpy()))))
                del cortes[k + 1]
                continue
            break
        return cortes

    # --- Interfaz fit / transform -----------------------------------------------
    def _asignar(self, serie: pd.Series, variable: str) -> pd.Series:
        cortes = self.cortes_[variable]
        etiquetas = _etiquetas(cortes, self.enteras_[variable])
        tramos = pd.cut(serie, bins=cortes, labels=etiquetas, right=True)
        return tramos.cat.add_categories(ETIQUETA_FALTANTE).fillna(ETIQUETA_FALTANTE)

    def fit(self, df: pd.DataFrame, variables: list[str]) -> "AgrupadorWoE":
        y = df[COLUMNA_OBJETIVO]
        for var in variables:
            x = df[var]
            self.enteras_[var] = bool((x.dropna() % 1 == 0).all())
            self.cortes_[var] = self._agrupar(x, y, monotonica=var not in self.no_monotonicas)
            orden = _etiquetas(self.cortes_[var], self.enteras_[var]) + [ETIQUETA_FALTANTE]
            tabla = tabla_woe(self._asignar(x, var), y, orden)

            # Faltantes con muy pocos clientes: sin evidencia suficiente -> riesgo promedio
            if (ETIQUETA_FALTANTE in tabla.index and
                    tabla.loc[ETIQUETA_FALTANTE, "clientes"] < self.min_clientes_faltante):
                tabla.loc[ETIQUETA_FALTANTE, ["woe", "iv"]] = 0.0
            self.tablas_[var] = tabla
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reemplaza cada variable por el WoE de su tramo.

        Si un tramo no existió en entrenamiento (p. ej., faltantes en una
        variable que no tenía faltantes), recibe WoE = 0: riesgo promedio.
        """
        salida = pd.DataFrame(index=df.index)
        for var, tabla in self.tablas_.items():
            tramos = self._asignar(df[var], var).astype(str)
            salida[var] = tramos.map(tabla["woe"]).fillna(0.0)
        if COLUMNA_OBJETIVO in df.columns:
            salida[COLUMNA_OBJETIVO] = df[COLUMNA_OBJETIVO]
        return salida

    def resumen_iv(self) -> pd.DataFrame:
        """IV total por variable, ordenado de mayor a menor."""
        filas = [{"variable": var, "iv": tabla["iv"].sum(), "tramos": len(tabla)}
                 for var, tabla in self.tablas_.items()]
        resumen = pd.DataFrame(filas).set_index("variable").sort_values("iv", ascending=False)
        resumen["poder_predictivo"] = resumen["iv"].map(poder_predictivo)
        return resumen


def seleccionar_variables(resumen_iv: pd.DataFrame, datos_woe: pd.DataFrame,
                          iv_minimo: float = IV_MINIMO,
                          correlacion_maxima: float = CORRELACION_MAXIMA) -> tuple[list[str], pd.DataFrame]:
    """Elige variables por IV y descarta las redundantes.

    1. Se descartan variables con IV menor a `iv_minimo`.
    2. Se recorren las demás de mayor a menor IV; si una tiene correlación
       mayor a `correlacion_maxima` con otra ya elegida, se descarta.
    """
    candidatas = resumen_iv[resumen_iv["iv"] >= iv_minimo].index.tolist()
    correlacion = datos_woe[candidatas].corr()

    seleccion, decisiones = [], []
    for var in resumen_iv.index:
        iv = resumen_iv.loc[var, "iv"]
        if iv < iv_minimo:
            decisiones.append((var, iv, "Descartada", f"IV menor a {iv_minimo}"))
            continue
        conflictos = [s for s in seleccion if abs(correlacion.loc[var, s]) > correlacion_maxima]
        if conflictos:
            s = conflictos[0]
            decisiones.append((var, iv, "Descartada",
                               f"Correlación {correlacion.loc[var, s]:.2f} con {s}"))
        else:
            seleccion.append(var)
            decisiones.append((var, iv, "Seleccionada", ""))

    tabla = pd.DataFrame(decisiones, columns=["variable", "iv", "decision", "motivo"]).set_index("variable")
    return seleccion, tabla


def graficar_iv(resumen_iv: pd.DataFrame, iv_minimo: float = IV_MINIMO):
    """Barras horizontales del IV por variable, con el umbral de selección."""
    datos = resumen_iv.sort_values("iv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(datos.index, datos["iv"], color="#2a78d6", height=0.6)
    ax.axvline(iv_minimo, color="#52514e", linestyle="--", linewidth=1)
    ax.annotate(f"Umbral IV = {iv_minimo}", xy=(iv_minimo, 0), xycoords=("data", "axes fraction"),
                xytext=(4, 4), textcoords="offset points", fontsize=8, color="#52514e")
    for posicion, valor in enumerate(datos["iv"]):
        ax.annotate(f"{valor:.3f}", xy=(valor, posicion), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Poder predictivo de cada variable (IV)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def preparar_woe() -> tuple[AgrupadorWoE, list[str], pd.DataFrame, pd.DataFrame]:
    """Ajusta el agrupador con entrenamiento, transforma ambas muestras y selecciona variables.

    Es determinista: en cualquier notebook posterior produce exactamente
    los mismos tramos, WoE y variables seleccionadas.
    """
    train, test = obtener_datos_limpios()
    agrupador = AgrupadorWoE(no_monotonicas=NO_MONOTONICAS).fit(train, VARIABLES_CANDIDATAS)
    train_woe, test_woe = agrupador.transform(train), agrupador.transform(test)
    seleccion, _ = seleccionar_variables(agrupador.resumen_iv(), train_woe)

    columnas = seleccion + [COLUMNA_OBJETIVO]
    train_woe[columnas].to_csv(ARCHIVO_TRAIN_WOE)
    test_woe[columnas].to_csv(ARCHIVO_TEST_WOE)
    return agrupador, seleccion, train_woe[columnas], test_woe[columnas]

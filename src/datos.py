"""Descarga, carga, validación y partición del dataset Give Me Some Credit."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARCHIVO_KAGGLE,
    ARCHIVO_TEST,
    ARCHIVO_TRAIN,
    COLUMNA_OBJETIVO,
    DIR_CRUDOS,
    PROPORCION_PRUEBA,
    RAIZ,
    SEMILLA,
)

# Nombres originales de Kaggle -> nombres en español (snake_case).
# El orden de este diccionario define el orden de las columnas del proyecto.
RENOMBRAR = {
    "SeriousDlqin2yrs": "malo",
    "RevolvingUtilizationOfUnsecuredLines": "uso_credito_revolvente",
    "age": "edad",
    "NumberOfTime30-59DaysPastDueNotWorse": "atrasos_30_59",
    "DebtRatio": "razon_deuda",
    "MonthlyIncome": "ingreso_mensual",
    "NumberOfOpenCreditLinesAndLoans": "num_creditos_abiertos",
    "NumberOfTimes90DaysLate": "atrasos_90",
    "NumberRealEstateLoansOrLines": "num_creditos_inmobiliarios",
    "NumberOfTime60-89DaysPastDueNotWorse": "atrasos_60_89",
    "NumberOfDependents": "num_dependientes",
}


def descargar_datos_kaggle(destino: Path = DIR_CRUDOS,
                           competencia: str = "GiveMeSomeCredit") -> None:
    """Descarga y descomprime los archivos de la competencia con la API de Kaggle.

    Requiere credenciales de Kaggle en variables de entorno y haber aceptado
    las reglas de la competencia en kaggle.com.
    """
    if ARCHIVO_KAGGLE.exists():
        print(f"Los datos ya existen en {ARCHIVO_KAGGLE.relative_to(RAIZ)}")
        return

    destino.mkdir(parents=True, exist_ok=True)
    try:
        resultado = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competencia, "-p", str(destino)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("No se encontró el cliente de Kaggle. Instálalo con: pip install kaggle")

    if resultado.returncode != 0:
        raise RuntimeError(
            "Falló la descarga desde Kaggle. Causas comunes: credenciales no "
            "configuradas o reglas de la competencia no aceptadas.\n"
            f"Detalle: {resultado.stderr or resultado.stdout}"
        )

    archivo_zip = destino / f"{competencia}.zip"
    with zipfile.ZipFile(archivo_zip) as zf:
        zf.extractall(destino)
    archivo_zip.unlink()  # Se borra el .zip para no duplicar espacio
    print("Archivos descargados:", sorted(p.name for p in destino.glob("*") if p.name != ".gitkeep"))


def copiar_datos_crudos(carpeta_origen: Path) -> None:
    """Copia cs-training.csv desde otra carpeta (p. ej., Google Drive) a data/raw/.

    Se trabaja sobre una copia local para que el resto del proyecto use
    siempre la misma ruta relativa, sin importar dónde se guardó el archivo.
    """
    if ARCHIVO_KAGGLE.exists():
        print(f"Los datos ya existen en {ARCHIVO_KAGGLE.relative_to(RAIZ)}")
        return

    origen = Path(carpeta_origen) / ARCHIVO_KAGGLE.name
    if not origen.exists():
        raise FileNotFoundError(
            f"No se encontró {origen}. Sube cs-training.csv a esa carpeta "
            "o ajusta la ruta (ver data/README.md)."
        )
    ARCHIVO_KAGGLE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, ARCHIVO_KAGGLE)
    print(f"Copiado {origen.name} a {ARCHIVO_KAGGLE.relative_to(RAIZ)}")


def cargar_datos_crudos(ruta: Path = ARCHIVO_KAGGLE) -> pd.DataFrame:
    """Lee cs-training.csv, valida sus columnas y las renombra al español."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta.name}. Revisa las instrucciones en data/README.md."
        )

    # La primera columna del CSV es un identificador sin nombre
    df = pd.read_csv(ruta, index_col=0)

    faltantes = set(RENOMBRAR) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en el archivo: {sorted(faltantes)}")

    df = df.rename(columns=RENOMBRAR)[list(RENOMBRAR.values())]
    df.index.name = "id_cliente"
    return df


def resumen_estructura(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla por variable: tipo de dato, nulos, valores únicos, mínimo y máximo."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "pct_nulos": df.isna().mean().mul(100).round(2),
        "valores_unicos": df.nunique(),
        "minimo": df.min(),
        "maximo": df.max(),
    })


def dividir_entrenamiento_prueba(df: pd.DataFrame,
                                 proporcion_prueba: float = PROPORCION_PRUEBA,
                                 semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide en entrenamiento y prueba, estratificando por la variable objetivo.

    La estratificación asegura que ambas muestras tengan la misma tasa de
    clientes malos, algo importante cuando la clase positiva es minoritaria.
    """
    train, test = train_test_split(
        df,
        test_size=proporcion_prueba,
        stratify=df[COLUMNA_OBJETIVO],
        random_state=semilla,
    )
    return train.sort_index(), test.sort_index()


def guardar_particiones(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Guarda las particiones en data/processed/ conservando el id del cliente."""
    ARCHIVO_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(ARCHIVO_TRAIN)
    test.to_csv(ARCHIVO_TEST)


def cargar_particiones() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las particiones guardadas en data/processed/."""
    train = pd.read_csv(ARCHIVO_TRAIN, index_col="id_cliente")
    test = pd.read_csv(ARCHIVO_TEST, index_col="id_cliente")
    return train, test


def obtener_particiones() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve entrenamiento y prueba; si no existen, las genera desde los datos crudos.

    Como la partición usa una semilla fija, regenerarla en una sesión nueva
    (por ejemplo, en Colab) produce exactamente las mismas filas.
    """
    if ARCHIVO_TRAIN.exists() and ARCHIVO_TEST.exists():
        return cargar_particiones()
    train, test = dividir_entrenamiento_prueba(cargar_datos_crudos())
    guardar_particiones(train, test)
    return train, test

# Datos

Este repositorio **no incluye el dataset**: los datos de la competencia están sujetos a las reglas de Kaggle y no deben redistribuirse. Aquí se explica cómo obtenerlos.

## Fuente

**Give Me Some Credit** — competencia de Kaggle  
https://www.kaggle.com/c/GiveMeSomeCredit

Solo se usa `cs-training.csv`. El archivo `cs-test.csv` no trae la variable objetivo (Kaggle la reservó para calificar la competencia), así que no sirve para medir el desempeño del modelo. Por eso la muestra de prueba se construye a partir de `cs-training.csv` (ver `notebooks/01_descarga_y_entorno.ipynb`).

## Opción A: Google Drive + Colab (la que usan los notebooks)

1. Crea una cuenta en kaggle.com, entra a la competencia y acepta sus reglas (pestaña *Rules*).
2. En la pestaña *Data*, descarga `cs-training.csv`.
3. Súbelo a Google Drive en `MyDrive/credit-risk-scorecard/cs-training.csv`.
4. Al ejecutar el notebook 01 en Colab se conecta Drive y el archivo se copia a `data/raw/`. Si lo guardaste en otra carpeta, cambia `CARPETA_DRIVE` en la primera celda.

## Opción B: API de Kaggle

Con credenciales de Kaggle configuradas como variables de entorno y el cliente instalado (`pip install kaggle`):

```python
from src.datos import descargar_datos_kaggle
descargar_datos_kaggle()  # Descarga y descomprime en data/raw/
```

## Opción C: en local

Descarga `cs-training.csv` desde la pestaña *Data* y colócalo en `data/raw/cs-training.csv`.

## Estructura esperada

```
data/
├── raw/
│   └── cs-training.csv     # Datos originales de Kaggle (no se suben)
└── processed/
    ├── train.csv           # 70 % entrenamiento (se genera en el notebook 01)
    └── test.csv            # 30 % prueba (se genera en el notebook 01)
```

## Diccionario de variables

| Nombre original (Kaggle) | Nombre en el proyecto | Descripción |
|---|---|---|
| SeriousDlqin2yrs | `malo` | **Variable objetivo.** 1 si el cliente tuvo 90 días o más de atraso en los siguientes 2 años. |
| RevolvingUtilizationOfUnsecuredLines | `uso_credito_revolvente` | Saldo en tarjetas y líneas personales sin garantía entre la suma de sus límites. |
| age | `edad` | Edad del cliente en años. |
| NumberOfTime30-59DaysPastDueNotWorse | `atrasos_30_59` | Veces con 30 a 59 días de atraso (sin llegar a más) en los últimos 2 años. |
| DebtRatio | `razon_deuda` | Pagos mensuales de deuda y gastos fijos entre el ingreso mensual bruto. |
| MonthlyIncome | `ingreso_mensual` | Ingreso mensual. |
| NumberOfOpenCreditLinesAndLoans | `num_creditos_abiertos` | Préstamos y líneas de crédito abiertos. |
| NumberOfTimes90DaysLate | `atrasos_90` | Veces con 90 días o más de atraso. |
| NumberRealEstateLoansOrLines | `num_creditos_inmobiliarios` | Préstamos hipotecarios o inmobiliarios. |
| NumberOfTime60-89DaysPastDueNotWorse | `atrasos_60_89` | Veces con 60 a 89 días de atraso (sin llegar a más) en los últimos 2 años. |
| NumberOfDependents | `num_dependientes` | Dependientes económicos, sin contar al titular. |

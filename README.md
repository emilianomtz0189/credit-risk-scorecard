# Modelo de riesgo de crédito: scorecard de probabilidad de incumplimiento

> 🚧 Proyecto en construcción. El resumen ejecutivo, los resultados y las conclusiones de negocio se agregan al terminar el modelo.

Modelo para estimar la **probabilidad de incumplimiento (PD)** de clientes de crédito al consumo con el dataset público *Give Me Some Credit* (Kaggle), siguiendo la metodología usada en bancos y financieras: agrupación de variables con WoE/IV, regresión logística convertida en scorecard de puntos, comparación con un modelo de *gradient boosting*, métricas de la industria (AUC, Gini, KS) y propuesta de puntos de corte con base en la pérdida esperada.

## Avance

| Etapa | Notebook | Estado |
|---|---|---|
| 1. Descarga de datos y preparación del entorno | `01_descarga_y_entorno.ipynb` | ✅ |
| 2. Exploración y limpieza | `02_exploracion_y_limpieza.ipynb` | ✅ |
| 3. Agrupación de variables, WoE e IV | `03_woe_iv.ipynb` | ✅ |
| 4. Regresión logística y scorecard | `04_scorecard.ipynb` | ✅ |
| 5. Modelo de comparación (gradient boosting) | `05_modelo_comparacion.ipynb` | 🔄 |
| 6. Métricas: AUC, Gini y KS | — | ⏳ |
| 7. Pérdida esperada y puntos de corte | — | ⏳ |

## Estructura del repositorio

```
├── data/          # Instrucciones de descarga (el dataset no se sube)
├── notebooks/     # Un notebook por etapa, numerados
├── src/           # Funciones reutilizables
└── requirements.txt
```

## Cómo reproducirlo

1. Sigue las instrucciones de `data/README.md` para obtener los datos.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta los notebooks en orden. Todo usa rutas relativas y una semilla fija (`SEMILLA = 42` en `src/config.py`).

Los notebooks están preparados para correr en Google Colab o en local.

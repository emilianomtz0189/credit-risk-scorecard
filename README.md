# Scorecard de riesgo de crédito: probabilidad de incumplimiento y puntos de corte

Modelo de **probabilidad de incumplimiento (PD)** para crédito al consumo, construido con la metodología que usan bancos y financieras: agrupación de variables con **WoE/IV**, **regresión logística** convertida en un **scorecard de puntos**, comparación contra un modelo de *machine learning* (**LightGBM**), métricas de la industria (**AUC, Gini, KS, PSI**) y una propuesta de **puntos de corte** basada en la **pérdida esperada** (PD × LGD × EAD).

Datos: [*Give Me Some Credit*](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle), 150,000 clientes con historial crediticio. Un cliente es **malo** si tuvo 90 días o más de atraso en los 2 años siguientes; la tasa de malos es de 6.68 %.

---

## Resumen ejecutivo

- **El scorecard separa muy bien a los clientes buenos de los malos:** en clientes que el modelo no vio al entrenar (muestra de prueba), obtiene un **AUC de 0.857, Gini de 0.714 y KS de 0.563**.
- **Revisando solo al 10 % de las solicitudes con menor puntaje se detecta el 55 % de los incumplimientos.**
- **Es explicable y estable:** cada decisión se justifica con los puntos de 8 variables; su AUC casi no cambia entre entrenamiento y prueba (0.858 vs. 0.857) y la población es estable (PSI de 0.0003).
- **Un modelo más complejo apenas lo supera:** LightGBM logra un AUC de 0.866, una mejora estadísticamente significativa pero pequeña (menos de 2 puntos de Gini), a cambio de perder explicabilidad y sobreajustar más. **Se recomienda el scorecard como modelo principal** y LightGBM como modelo retador.
- **Estrategia de decisión propuesta:** rechazar debajo de 520 puntos, revisión manual de 520 a 567 y aprobación automática desde 568. Rechazando solo al **5.7 %** de las solicitudes se evita el **39 %** de los incumplimientos; con los supuestos de negocio usados, la pérdida baja **39 %** y la utilidad sube **$16.2 millones** frente a aprobar a todos.

---

## Resultados principales

### Desempeño de los modelos (muestra de prueba, 45,000 clientes)

| Métrica | Scorecard | LightGBM |
|---|---|---|
| AUC | 0.8569 | 0.8661 |
| Gini | 0.7137 | 0.7323 |
| KS | 0.5629 | 0.5777 |
| Caída de AUC de entrenamiento a prueba | 0.0015 | 0.0113 |

Diferencia de AUC (LightGBM − scorecard): 0.0093, con intervalo de confianza de 95 % de 0.0074 a 0.0114 (*bootstrap*).

### Concentración del riesgo (scorecard)

| % de solicitudes con menor puntaje | % de todos los incumplimientos detectados |
|---|---|
| 10 % | 54.55 % |
| 20 % | 71.51 % |
| 30 % | 81.22 % |

### Variables del scorecard

| Variable | IV | Rango de puntos |
|---|---|---|
| Uso de crédito revolvente | 1.13 | 59 |
| Atrasos de 90+ días | 0.85 | 49 |
| Atrasos de 30–59 días | 0.74 | 47 |
| Atrasos de 60–89 días | 0.58 | 35 |
| Razón de deuda | 0.08 | 23 |
| Edad | 0.24 | 18 |
| Créditos inmobiliarios | 0.06 | 18 |
| Dependientes | 0.04 | 7 |

Escala: 600 puntos equivalen a 50 clientes buenos por cada malo, y cada 20 puntos (PDO) los momios se duplican. Más puntos = menos riesgo. Ingreso mensual y número de créditos abiertos se descartaron por no ser significativos junto con las demás variables.

### Estrategia de puntos de corte (resultados reales en prueba)

| Zona | Puntaje | % de solicitudes | Tasa de malos |
|---|---|---|---|
| Aprobación automática | 568 o más | 73.50 % | 1.97 % |
| Revisión manual | 520 a 567 | 20.78 % | 12.53 % |
| Rechazo | Menos de 520 | 5.72 % | 46.02 % |

| Escenario | % aprobado | Tasa de malos | Pérdida | Utilidad |
|---|---|---|---|---|
| Aprobar a todos | 100.00 % | 6.68 % | $67.7 M | $247.3 M |
| **Estrategia propuesta** (automática + revisión) | **94.28 %** | **4.30 %** | **$41.0 M** | **$263.5 M** |
| Solo aprobación automática | 73.50 % | 1.97 % | $14.7 M | $228.5 M |

Supuestos (el dataset no trae montos ni recuperaciones): exposición de $50,000 por cliente, LGD de 45 % (referencia de Basilea para deuda sin garantía) y margen de 15 % en el horizonte de 2 años. Con LGD de 35 % a 85 % y margen de 10 % a 25 %, el corte de rechazo se mantiene entre 500 y 560 puntos.

---

## Conclusiones de negocio

1. **El historial de pago y el uso de la línea de crédito explican la mayor parte del riesgo.** Las tres variables de atrasos y el uso de crédito concentran el poder predictivo, tanto en el scorecard como en LightGBM.
2. **Los datos "raros" son información, no basura.** Los códigos 96/98 en atrasos identificaron a 182 clientes con 54 % de incumplimiento, y los clientes sin ingreso reportado resultaron menos riesgosos que el promedio. Ambos se conservaron como grupos propios.
3. **La explicabilidad cuesta poco.** El scorecard ordena el riesgo casi igual que LightGBM (correlación de −0.95; coinciden en el 88 % de los clientes más riesgosos) y es más robusto.
4. **Ser demasiado estricto también destruye valor.** La zona de revisión manual tiene 12.5 % de malos, pero es rentable en promedio; rechazarla completa reduciría la utilidad. Debe revisarse para depurarla, no rechazarse.

## Limitaciones y siguientes pasos

- **Sin fechas:** no se puede validar el modelo en un periodo posterior (*out-of-time*); en producción, el PSI debe monitorearse cada mes.
- **Sin montos ni recuperaciones:** LGD, EAD y margen son supuestos; deben sustituirse por datos reales de la institución.
- **Solo clientes aprobados:** el dataset no incluye solicitudes rechazadas, lo que puede sesgar el modelo (*reject inference*).
- **Variables adicionales útiles con datos reales:** porcentaje de pago del saldo, uso promedio de varios meses y antigüedad de las cuentas.

---

## Metodología

| Etapa | Notebook | Qué se hace |
|---|---|---|
| 1 | [`01_descarga_y_entorno`](notebooks/01_descarga_y_entorno.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/01_descarga_y_entorno.ipynb) | Carga y validación de datos; partición 70/30 estratificada antes de explorar. |
| 2 | [`02_exploracion_y_limpieza`](notebooks/02_exploracion_y_limpieza.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/02_exploracion_y_limpieza.ipynb) | Faltantes informativos, códigos especiales, valores imposibles y topes aprendidos solo con entrenamiento. |
| 3 | [`03_woe_iv`](notebooks/03_woe_iv.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/03_woe_iv.ipynb) | Agrupación con tendencia monótona, WoE, IV y selección de variables. |
| 4 | [`04_scorecard`](notebooks/04_scorecard.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/04_scorecard.ipynb) | Regresión logística, conversión a puntos y explicación por cliente. |
| 5 | [`05_modelo_comparacion`](notebooks/05_modelo_comparacion.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/05_modelo_comparacion.ipynb) | LightGBM con validación cruzada; importancia de variables y SHAP. |
| 6 | [`06_metricas`](notebooks/06_metricas.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/06_metricas.ipynb) | AUC, Gini, KS, bootstrap, tabla de deciles y PSI. |
| 7 | [`07_perdida_esperada`](notebooks/07_perdida_esperada.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/emilianomtz0189/credit-risk-scorecard/blob/main/notebooks/07_perdida_esperada.ipynb) | Calibración de PD, pérdida esperada, puntos de corte y sensibilidad. |

## Estructura del repositorio

```
├── data/                  # Instrucciones de descarga (el dataset no se sube)
├── notebooks/             # Un notebook por etapa, numerados
├── src/
│   ├── config.py          # Rutas relativas, semilla y parámetros globales
│   ├── datos.py           # Descarga, carga, validación y partición
│   ├── limpieza.py        # Reglas de limpieza (fit/transform)
│   ├── eda.py             # Análisis exploratorio
│   ├── woe.py             # Agrupación, WoE, IV y selección de variables
│   ├── scorecard.py       # Regresión logística y tabla de puntos
│   ├── comparacion.py     # Modelo LightGBM
│   ├── metricas.py        # AUC, Gini, KS, deciles y PSI
│   └── perdida.py         # Pérdida esperada y puntos de corte
└── requirements.txt       # Versiones fijas de las librerías
```

## Cómo reproducirlo

1. Descarga `cs-training.csv` siguiendo [`data/README.md`](data/README.md).
2. Abre cualquier notebook en Google Colab con su enlace de la tabla de metodología; cada uno conecta Drive, clona el repositorio e instala `requirements.txt`. En local: `pip install -r requirements.txt` y coloca el archivo en `data/raw/`.
3. Todo es determinista: rutas relativas, semilla fija (`SEMILLA = 42`) y versiones de librerías fijas. Cada notebook reconstruye lo que necesita de las etapas anteriores.

**Herramientas:** Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, Matplotlib, Google Colab.

## Autor

**José Emiliano Martínez Chavira** · Actuario (UAG, Actuaría y Ciencia de Datos) · [LinkedIn](https://www.linkedin.com/in/emiliano-martinez-861175227)

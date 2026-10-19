# Problem framing: triaje de enfermedades autoinmunes

Documento de definición del problema para el proyecto. Vincula la actividad de
identificación de oportunidades de ML con el dataset y los notebooks del repositorio.

## 1. Problema identificado

En atención primaria, el triaje de pacientes con síntomas inespecíficos —fatiga
persistente, dolor articular, erupciones, sequedad ocular, fiebre baja— se apoya en
reglas empíricas y en la intuición del clínico. El resultado es heterogéneo: algunos
pacientes esperan meses para una evaluación especializada, mientras otros acumulan
pruebas innecesarias.

La oportunidad de ML es construir un sistema de apoyo que priorice a los pacientes con
mayor probabilidad de enfermedad autoinmune y oriente la derivación al especialista
adecuado. El modelo no reemplaza el juicio médico: ordena la cola de evaluación y
ayuda a decidir dónde concentrar los recursos.

## 2. Datos necesarios

El proyecto ya dispone de un dataset limpio con **12.500 registros**, **77 variables** y
**116 clases** en `Diagnosis` (2.500 `Normal` y 10.000 casos autoinmunes repartidos en
115 enfermedades). Las variables mínimas para el caso son:

| Variable | Columna(s) del dataset | Tipo | Rol |
|---|---|---|---|
| Edad | `Age` | Numérica | Variable demográfica y de ajuste. |
| Sexo | `Gender` | Categórica | Muchas enfermedades autoinmunes tienen prevalencia desigual. |
| Duración de síntomas | `Sickness_Duration_Months` | Numérica | Cronicidad; **excluir de la etapa binaria** por fuga de información. |
| Inflamación | `CRP`, `ESR` | Numéricas | Actividad inflamatoria sistémica. |
| Hemograma | `Hemoglobin`, `WBC_Count`, `PLT_Count`, `RBC_Count` | Numéricas | Anemia, infección y respuesta inmune. |
| Anticuerpos | `ANA`, `Anti-dsDNA`, `Anti-TPO`, `Anti-Tg`, `Anti-SMA`, etc. | Binarias | Especificidad diagnóstica por subtipo. |
| Síntomas | `Fatigue or chronic tiredness`, `Joint pain`, `Rashes and skin lesions`, `Low-grade fever`, etc. | Binarias | Patrón clínico que motiva la consulta. |

Para que el modelo aprenda el patrón se necesitan registros históricos con diagnóstico
confirmado y las mismas variables disponibles en el momento del triaje. No sirve una
variable que solo se conoce después del diagnóstico.

## 3. Objetivo y tipo de aprendizaje

**Tipo:** aprendizaje supervisado de clasificación.

- **Etapa 1 — triaje:** clasificación binaria `Normal` vs. `Autoinmune`. La salida es
  una probabilidad que permite priorizar la derivación.
- **Etapa 2 — routing:** clasificación multiclase entre las 115 enfermedades
  autoinmunes. La salida orienta al especialista (reumatología, endocrinología,
  dermatología, etc.).

**Etiqueta:** `Diagnosis`.

**Métricas:** recall o sensibilidad de la clase autoinmune (no perder pacientes
enfermos), F1-macro (equidad entre clases), calibración de probabilidades y validación
externa. El split debe ser estratificado por clase; la exactitud global no es una
métrica útil con este desbalance.

## 4. Sesgos y riesgos

- **Desbalance 40:1.** `Normal` concentra 2.500 casos y las clases minoritarias tienen
  entre 62 y 184. Un modelo que siempre prediga la clase mayoritaria puede parecer
  razonable en accuracy y ser inútil en la práctica.
- **Missingness informativa.** Varias columnas de anticuerpos (`Anti-dsDNA`, `Anti-Sm`,
  `Rheumatoid factor`, `ACPA`, `Anti-TPO`, `Anti-Tg`, `Anti-SMA`) se rellenaron con 0.
  Todos los casos `Normal` quedan en 0, mientras que en autoinmunes hay una mezcla de 0 y
  1. El modelo puede aprender que "no hay anticuerpos" equivale a "sano", aunque en la
  práctica la prueba no se haya solicitado. Es un caso de *Garbage In, Garbage Out*.
- **Fuga de información.** `Sickness_Duration_Months` vale 0 en todos los `Normal` y
  entre 1 y 119 en autoinmunes. Separa los grupos por construcción; no debe usarse en el
  modelo binario.
- **Sesgo de fuente y de etiqueta.** El dataset proviene de una sola fuente y solo
  incluye pacientes que llegaron a un diagnóstico. La distribución no representa
  necesariamente a la población que se atiende en un centro de salud.
- **Sesgo demográfico.** Edad y sexo pueden correlacionar con la enfermedad y con el
  acceso al diagnóstico. Conviene auditar métricas por subgrupo.
- **Correlación no causal.** La EDA visual mostró que la correlación global PCR–VSG
  (r = 0,40) desaparece dentro de cada grupo (r ≈ 0,03). Es un efecto de mezclar sanos y
  enfermos, no una relación causal.

## 5. Criterios de éxito

1. Un baseline binario reproducible con split estratificado y recall alto para la clase
   autoinmune.
2. Un modelo multiclase con F1-macro razonable y calibración aceptable.
3. Un informe de equidad por sexo y grupo de edad.
4. Validación en una muestra externa o, como mínimo, en un hold-out no usado durante el
   desarrollo.

## 6. Relación con el repositorio

- Dataset limpio: `data/processed/autoimmune_cleaned.parquet`.
- EDA y estructura de clases: `notebooks/01_eda.ipynb`.
- EDA visual, correlaciones y fuga de información: `notebooks/03_eda_visual.ipynb`.
- Comparación de modelos, umbral y métricas: `notebooks/04_model_evaluation.ipynb`.
- Siguiente paso: baseline `Normal` vs. `Autoinmune` sin `Sickness_Duration_Months`,
  con métricas macro y análisis de calibración.

El análisis de caso de una página se entrega como PDF fuera del repositorio; este
documento es la versión versionada y ampliada del mismo planteamiento.

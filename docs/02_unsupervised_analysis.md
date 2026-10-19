# Análisis de caso: aprendizaje no supervisado

Documento de reflexión del módulo de Aprendizaje No Supervisado. El caso plantea
segmentar **500.000 usuarios** de una app de streaming musical con **100 variables**
(géneros escuchados, horas de escucha al día, playlists creadas, edad y ubicación) y
**sin etiquetas**. La versión aplicada sobre el dataset del proyecto vive en
`notebooks/05_unsupervised_learning.ipynb`.

## 1. Estrategia de clustering

La elección depende de dos preguntas: qué forma tienen los grupos y cómo debe tratar el
algoritmo al ruido.

| Criterio | K-Means | DBSCAN |
|---|---|---|
| Forma de los grupos | Convexa y aproximadamente esférica; cada punto pertenece a un centroide. | Arbitraria: sigue regiones de alta densidad, incluso con formas alargadas o anillos. |
| Manejo del ruido | No distingue ruido: todo punto se asigna a un cluster y los outliers deforman los centroides. | Marca explícitamente el ruido con la etiqueta -1; es su gran ventaja. |
| Escalabilidad | Muy alto: `MiniBatchKMeans` procesa millones de filas en lotes. | Menor: la búsqueda de vecinos se degrada en alta dimensión y con densidades dispares. |
| Parámetros | Solo `k`, elegido con codo, silueta o Davies-Bouldin. | `eps` y `min_samples`, difíciles de fijar y sensibles a la escala. |
| Interpretabilidad | Alta: los centroides y perfiles por variable se comunican fácil a Marketing. | Media: grupos irregulares y ruido fuera del análisis. |
| Cuándo usarlo | Segmentación estable y accionable de toda la base. | Detección de anomalías: bots, cuentas compartidas, patrones extremos. |

**Elección: K-Means como motor principal y DBSCAN como detector de anomalías.** Con
500.000 usuarios y 100 variables, la densidad pierde significado: en dimensiones altas
casi todos los puntos quedan lejos entre sí y DBSCAN tiende a marcar demasiado ruido o a
unir todo en un solo grupo. K-Means escala bien, asigna todos los usuarios a un segmento
y produce centroides interpretables. Su contraparte es la sensibilidad al ruido: por eso
conviene detectar primero los valores atípicos con DBSCAN (o un modelo de aislamiento) y
segmentar después sobre la base depurada.

## 2. Reducción de dimensionalidad con PCA

Con 100 variables por usuario, la distancia entre puntos se vuelve menos informativa
(maldición de la dimensionalidad) y el clustering se vuelve más lento y más ruidoso. PCA
se aplica **antes** del clustering, después de codificar las variables categóricas y
estandarizar todo, porque PCA es sensible a la escala.

Procedimiento:

1. Codificar géneros y ubicación con one-hot o embeddings.
2. Estandarizar las variables numéricas.
3. Ajustar PCA y graficar la varianza explicada acumulada.
4. Retener suficientes componentes para explicar entre 80% y 90% de la varianza.
5. Usar 2-3 componentes para visualizar y 15-25 para clusterizar.
6. Comparar la silueta en el espacio original y en el espacio PCA.

Beneficios: reduce ruido y multicolinealidad; acelera las distancias y permite probar
muchos valores de `k`; hace posible visualizar a los usuarios en un plano interpretable;
ayuda a estimar cuántas dimensiones reales tiene la variación de la base.

Precauciones: los componentes de PCA son mezclas lineales de las variables originales, no
variables interpretables; PCA es lineal y puede ocultar estructuras no lineales. Para
visualización no lineal se pueden explorar t-SNE o UMAP, pero conviene clusterizar en el
espacio PCA.

## 3. Interpretación de negocio: ponerles nombre a los 5 grupos

El algoritmo devuelve grupos, pero un grupo sin nombre no sirve para Marketing. El
proceso es humano y sistemático:

1. **Perfilar cada grupo.** Medias, medianas y modas de las variables clave (géneros
   dominantes, horas de escucha, playlists, edad, ubicación) contra el promedio general.
2. **Verificar que los grupos son distintos.** Pruebas estadísticas (ANOVA, chi-cuadrado)
   y tamaños de efecto; si dos grupos son casi idénticos, conviene reducir `k`.
3. **Comprobar estabilidad.** Repetir K-Means con varias semillas y submuestras; si los
   grupos cambian demasiado, la segmentación no es confiable.
4. **Nombrar con el equipo.** Taller con Marketing y Producto para convertir perfiles
   numéricos en personas: "oyentes de fondo", "curadores de listas", "exploradores
   nocturnos", "fieles a un género", "usuarios esporádicos".
5. **Definir acciones.** Para cada segmento, una acción concreta (recomendaciones,
   playlist editorial, promoción, reactivación) y una métrica (retención, conversión a
   premium, horas escuchadas).
6. **Medir y revisar.** Los nombres son hipótesis: validarlos con A/B tests y volver a
   perfilar cada trimestre.

Un segmento útil debe ser **distinguible**, **suficientemente grande**, **estable** y
**accionable**. Si un grupo no permite una decisión distinta a la de otro, no aporta
valor aunque la silueta sea alta.

## 4. Ética y sesgos

Agrupar usuarios sin supervisión no elimina los sesgos: los hereda de los datos y del
diseño. Un riesgo central es usar edad y ubicación como *proxies* de nivel socioeconómico,
cultura o etnia. Si los grupos deciden precios, recomendaciones o promociones, el modelo
puede reproducir desigualdades y hasta agravarlas.

- **Sesgo de representación.** La base solo incluye usuarios de la app; quienes no usan
  streaming o lo hacen desde dispositivos limitados quedan fuera.
- **Sesgo por variables proxy.** Ubicación y edad pueden correlacionar con características
  protegidas aunque el algoritmo no las vea directamente.
- **Bucle de retroalimentación.** Recomendar música solo a ciertos grupos hace que sus
  datos futuros reflejen esas recomendaciones y el sesgo se refuerza.
- **Inferencias sensibles.** Los patrones de escucha pueden revelar estado de ánimo, salud
  o preferencias íntimas; agrupar puede convertirse en vigilancia.

**Mitigación.** Auditar la composición de cada grupo por variables demográficas; evitar
que atributos protegidos o sus proxies definan precios y acceso; agregar o anonimizar
antes de compartir segmentos; documentar limitaciones y usos prohibidos; mantener revisión
humana; incluir diversidad en el equipo que nombra los grupos; y medir el impacto dispar
antes y después de lanzar una acción.

## 5. Síntesis

Para segmentar 500.000 usuarios: estandarizar y codificar, aplicar PCA, elegir `k` con
codo y silueta, segmentar con K-Means y usar DBSCAN como detector complementario de
anomalías. El resultado no es una verdad: es el comienzo de una investigación. El valor
aparece cuando el equipo de negocio convierte los grupos en decisiones y cuando la
organización audita los sesgos antes de actuar.

## Aplicación en el repositorio

`notebooks/05_unsupervised_learning.ipynb` aplica estas ideas al dataset del proyecto
(12.500 pacientes). Excluye `Sickness_Duration_Months` por fuga de información, reduce con
PCA, segmenta con K-Means y usa DBSCAN para detectar atípicos. Los grupos se comparan con
`Diagnosis` **solo después** del ajuste, a modo interpretativo. Las funciones reutilizables
viven en `src/autoimmune_disorder_prediction/clustering.py`.

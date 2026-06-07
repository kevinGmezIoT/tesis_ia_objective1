# Resumen para Presentación (PPT - 10 Diapositivas)

Este documento contiene los puntos clave estructurados para una presentación de 10 diapositivas sobre el artículo:  
**"Construcción y caracterización de un corpus multicámara de secuencias RGB–silueta para Re-ID de personas bajo similitud interclase, oclusión e iluminación variable"**

---

## 📋 Estructura de las Diapositivas

### Diapositiva 1: Título y Contexto
* **Título de la presentación**: Construcción y Caracterización de un Corpus Multimodal para Re-Identificación de Personas (Re-ID).
* **Propósito**: Presentar un nuevo conjunto de datos diseñado para someter a pruebas de estrés a los algoritmos de Re-ID y Gait Re-ID bajo condiciones extremas y realistas.
* **Modalidad dual**: Integración sincronizada de imágenes en color (**RGB**) y formas binarias (**siluetas**).
* **Problema a resolver**: Las limitaciones de los actuales sistemas basados únicamente en apariencia que fallan ante uniformes o ropa similar y entornos no controlados.

---

### Diapositiva 2: Planteamiento del Problema
* **Retos en Re-ID del mundo real**:
  * **Variación intraclase**: Cambios drásticos de iluminación, oclusión intensa y mala calidad en las siluetas extraídas.
  * **Ambigüedad interclase**: Diferentes personas que llevan vestimentas idénticas o muy similares (e.g., uniformes, ropa de trabajo).
* **Deficiencia en la literatura actual**:
  * Los conjuntos de datos públicos existentes (e.g., MARS, iLIDS-VID) tienen una alta separabilidad de clases y carecen de anotaciones detalladas y simultáneas de estas adversidades.
* **Hipótesis**: Un corpus construido explícitamente con estas dificultades presentará un margen de separabilidad mucho menor y revelará las fallas reales de los modelos.

---

### Diapositiva 3: La Propuesta: Estructura del Corpus Multimodal
* **Características principales**:
  * Secuencias de peatones sincronizadas en **RGB** y **Siluetas binarias** a través de múltiples cámaras.
  * Agrupación periódica en **Ciclos de marcha** (morfología de locomoción) en lugar de ventanas de tiempo fijas.
* **Jerarquía de Información (4 niveles)**:
  1. **Fotograma**: Recortes de imagen, siluetas y coordenadas de cuadros delimitadores ($B_{i,t}$).
  2. **Secuencia**: Iluminación, oclusión, calidad de silueta y descriptores del ciclo de marcha.
  3. **Identidad/Sesión**: Atributos semánticos de la vestimenta consolidados.
  4. **Corpus**: Matrices de distancia de Gower, métricas de separabilidad y dificultad.

---

### Diapositiva 4: Metodología I: Ciclos de Marcha y Calidad de Silueta
* **Segmentación de Ciclos de Marcha**:
  * Transformación de siluetas en una señal temporal 1D mediante suma del primer plano ($x_i(t)$).
  * Estimación del periodo dominante del paso ($T_i$) mediante la función de autocorrelación normalizada.
  * Duración estimada del ciclo completo de marcha: $L_i^{cycle} \approx 2T_i$.
* **Evaluación de la Calidad de Silueta (SQQ)**:
  * Cálculo de la energía de la marcha estructurada frente a la energía del ruido ($Q_i(r) = \sigma^2_{s,i}(r) - \sigma^2_{z,i}(r)$).
  * Filtrado morfológico (QAGait) para descartar ruido de fondo y formas no humanas.

---

### Diapositiva 5: Metodología II: Iluminación y Oclusión Objetiva
* **Caracterización de la Iluminación (RGB)**:
  * Luminancia media calculada bajo el estándar BT.601 ($Y = 0.299R + 0.587G + 0.114B$).
  * Clasificación ordinal en 4 niveles (baja, medio-baja, medio-alta, alta) calibrada frente a un corpus común junto con MARS e iLIDS-VID.
* **Clasificación de Oclusión (Método de Gilroy)**:
  * Detección de 17 puntos clave mediante Faster R-CNN (ResNet-50-FPN).
  * Estimación en 2D de la superficie corporal visible (BSA) dividida en 11 regiones anatómicas.
  * Niveles de oclusión: **Baja** ($<40\%$), **Parcial** ($40\% - 80\%$) e **Intensa** ($>80\%$).

---

### Diapositiva 6: Metodología III: Similitud Híbrida y Marco de Ambigüedad
* **Similitud Semántica Discreta (Atributos)**:
  * Coeficiente de Gower aplicado a 5 atributos clave: prenda superior (tipo/color), prenda inferior (tipo/color) y presencia de mochila.
* **Similitud Visual Continua (Incrustaciones Profundas)**:
  * Embeddings extraídos mediante la arquitectura **OSNet** (Omni-Scale Network).
  * Generación de **prototipos de identidad equilibrados por cámara** para mitigar sesgos de dominio y de iluminación.
* **Cuadrantes de Ambigüedad**:
  * Cruce de percentiles comunes ($P_{90}^{Gower}$ y $P_{90}^{OSNet}$) para clasificar los pares de identidades en 4 categorías de ambigüedad.

---

### Diapositiva 7: Protocolo de Evaluación Comparativa
* **Comparación con 4 Datasets de Referencia**:
  * **CASIA-B** y **Gait3D** para la comparación basada en siluetas y análisis de marcha.
  * **iLIDS-VID** y **MARS** para la comparación de secuencias y metadatos RGB.
* **Métrica clave**: **Índice de separabilidad relativa** ($R_{sep}$):
  $$R_{sep} = \frac{\bar{d}_{inter}}{\bar{d}_{intra}}$$
  * Un índice más bajo indica mayor dificultad del conjunto de datos (menor distancia entre identidades diferentes en relación a la misma identidad).

---

### Diapositiva 8: Resultados I: Nivel de Dificultad Macro (RGB)
* **Comparación de Separabilidad**:
  * **Corpus propuesto**: Similitud intraclase = **0.629** | Ratio de separabilidad = **1.46** | Ratio entre cámaras = **1.33**
  * **iLIDS-VID**: Similitud intraclase = **0.797** | Ratio de separabilidad = **1.77** | Ratio entre cámaras = **1.82**
  * **MARS**: Similitud intraclase = **0.782** | Ratio de separabilidad = **2.18** | Ratio entre cámaras = **2.01**
* **Conclusión**: El corpus propuesto tiene el ratio de separabilidad más bajo (1.46), lo que valida cuantitativamente que es el benchmark más difícil y realista de la terna.

---

### Diapositiva 9: Resultados II: Análisis de Cuadrantes de Ambigüedad
* **Clasificación de los 2,701 pares de identidades**:
  * **Baja ambigüedad (73.4%)**: Parejas distintas y fácilmente separables.
  * **Similitud semántica controlable (16.5%)**: Comparten ropa/mochila, pero OSNet las distingue con éxito mediante micro-características.
  * **Ambigüedad confirmada (3.3% - 90 pares)**: Comparten ropa y OSNet falla rotundamente al separarlos. Representan el principal reto de diseño.
  * **Ambigüedad visual inexplicable (6.7% - 181 pares)**: Vestimentas diferentes, pero OSNet los agrupa erróneamente.
* **Impacto ambiental**: Las condiciones adversas (baja iluminación, oclusión y siluetas degradadas) aumentan drásticamente la tasa de falsos positivos en el cuadrante de ambigüedad confirmada.

---

### Diapositiva 10: Conclusiones y Trabajo Futuro
* **Conclusiones**:
  * Los actuales modelos de Re-ID basados en RGB dependen críticamente de la apariencia macro (color y tipo de prenda), lo que los hace vulnerables en escenarios reales no controlados o con uniformes.
  * El nuevo corpus multimodal expone esta vulnerabilidad al aislar con precisión los fallos del modelo.
* **Trabajo Futuro**:
  * **Fusión Multimodal**: Combinar la apariencia RGB con la locomoción temporal (ciclos de marcha de siluetas) para resolver la ambigüedad de ropa similar.
  * **Expansión**: Integrar topología de red de cámaras (análisis espacio-temporal) y estimación 3D del esqueleto humano.

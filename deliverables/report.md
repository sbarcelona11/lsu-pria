# Informe técnico — {{TITLE}}

**Grupo:** {{GROUP}}  
**Integrantes:** {{MEMBERS}}  
**Curso:** {{COURSE}}  
**Fecha:** {{DATE}}  

## 1. Introducción

El proyecto aborda la traducción inicial de Lengua de Señas Uruguaya (LSU) a texto (y apoyo por voz) a partir de video. La motivación principal es explorar una aplicación de Visión Computacional vinculada a accesibilidad e interacción persona-computadora, priorizando un alcance realizable dentro del curso.

En lugar de intentar resolver traducción libre completa, el trabajo se enfoca en un escenario reproducible: tomar segmentos de video (clips) donde aparece el intérprete y devolver una hipótesis textual para cada segmento. Esto permite construir una demo funcional de “texto y voz” y reportar métricas comparables, además de discutir limitaciones reales del alineamiento texto‑señas.

El proyecto usa el dataset **iLSU‑T** (TV broadcasting) como fuente principal de datos, y una estrategia de supervisión basada en transcripciones **WhisperX** por segmento (clip → texto).

## 2. Referencial teórico

El sistema se apoya en cuatro ideas centrales. Primero, OpenCV permite realizar procesamiento de video (lectura, muestreo temporal, preprocesado y visualización) de forma eficiente. Segundo, MediaPipe provee detectores robustos para manos y cuerpo, que permiten extraer representaciones compactas (landmarks y ROI) y desacoplar parcialmente el gesto del fondo de la escena.

Tercero, se usa el concepto de **SLT (Sign Language Translation)**: traducir una secuencia visual (video) a una secuencia textual. En la práctica, los modelos modernos requieren (a) segmentación temporal, (b) extracción de features por frame/clip y (c) un modelo secuencial (por ejemplo, Transformer) para mapear secuencias a texto. En esta entrega se adopta una aproximación pragmática: construir un dataset por segmentos a partir de WhisperX y evaluar un baseline reproducible.

Por último, se discute el rol del alineamiento: WhisperX produce texto aproximado (y tiempos) que no siempre coincide exactamente con el contenido signado. Esto introduce ruido supervisado que afecta el techo de rendimiento y obliga a evaluar con métricas tolerantes a errores parciales (además de inspección cualitativa).

## 3. Metodología

### 3.1 Datos

En esta entrega no se utiliza un dataset propio recolectado con webcam. La fuente de datos utilizada es **iLSU‑T** (sources `source2` y `source3`), que consiste en material de TV con video y transcripciones WhisperX.

La estructura de datos relevante para reproducir los experimentos es:
- `data/ilsut_extracted/source2/episodes/` y `data/ilsut_extracted/source3/episodes/` (videos)
- `data/ilsut_extracted/source2/whisperx/` y `data/ilsut_extracted/source3/whisperx/` (JSON WhisperX con segmentos)
- artefactos generados por el pipeline bajo `runs/.../` (subset/export/train/eval y `summary.json`)

En iLSU‑T no se parte de etiquetas frame‑a‑frame sino de segmentos de texto (WhisperX) alineados débilmente al video. A partir de esos segmentos se construye un manifest temporal (clip → texto) y se extraen features/ROI para entrenar y evaluar modelos de traducción (SLT).

### 3.2 Herramientas

Las herramientas principales del proyecto son Python, OpenCV, MediaPipe Hands, scikit-learn y PyTorch. Para la interfaz de usuario se implementó una aplicación web con FastAPI en backend y React (Vite) en frontend, capaz de capturar webcam, enviar frames al backend, mostrar la predicción y componer frases simples.

### 3.3 Pipeline de procesamiento

El flujo de SLT (clip → texto) implementado es el siguiente:

1. A partir de iLSU‑T, se leen videos y WhisperX JSON para obtener segmentos temporales (start/end) con texto.
2. Se filtran segmentos por duración y longitud de texto para estabilizar el set de entrenamiento/demostración.
3. Se construye un manifest reproducible con `group_id` por episodio y splits (train/val/test) sin leakage por episodio.
4. Se recortan clips por segmento (opcional) y se extraen features visuales por clip muestreando frames a `sample_fps`.
5. Se entrena un baseline proxy (KNN sobre embeddings agregados) como referencia reproducible y se evalúa en test.
6. Opcionalmente, se empaquetan los splits en formato SignJoey y se entrena un modelo generativo (Transformer) usando `neccam/slt` en GPU.
7. La demo web permite analizar un video offline (subido o URL), obtener texto predicho y reproducirlo por TTS.

### 3.4 Entrenamiento

Se entrenan dos variantes complementarias para SLT:

1) **Baseline proxy (reproducible y rápido)**  
Se extraen features por clip (secuencia de vectores) y se agregan en un embedding fijo por muestra. Sobre ese embedding se entrena un KNN que predice directamente el texto del segmento. Este baseline no intenta generalizar lingüísticamente, pero permite:
- validar el pipeline end‑to‑end,
- medir impacto de filtros y extracción de features,
- y sostener una demo confiable sin requerir GPU/CUDA.

2) **Modelo generativo (opcional)**  
Se exporta el dataset a formato SignJoey y se entrena un Transformer (SignJoey) que traduce secuencias a texto. Este enfoque es más apropiado para “traducción real”, pero requiere GPU/CUDA y dependencias extra; por eso se recomienda correrlo en Colab.

## 4. Resultados y discusión

Los resultados se reportan para el problema de SLT (clip → texto) construido a partir de iLSU‑T + WhisperX. Además de la métrica exacta (match completo), se reportan métricas que capturan aciertos parciales a nivel de tokens.

En discusión, interesa interpretar los errores desde dos fuentes principales:
- **Ruido de supervisión/alineamiento**: WhisperX no siempre coincide con lo signado y puede estar desfasado temporalmente.
- **Información visual incompleta**: LSU no depende solo de la mano; también importan cara, pose, movimiento y contexto discursivo.

{{SLT_REPORT_SECTION}}

**Figura sugerida:** tabla/plot de métricas (proxy vs generativo) + ejemplos cualitativos de errores frecuentes.

## 5. Conclusión

El proyecto entrega un pipeline reproducible para SLT con iLSU‑T + WhisperX, una demo web “texto y voz”, y un esquema de evaluación con métricas cuantitativas + análisis cualitativo. El baseline proxy permite validar la viabilidad técnica con bajo costo computacional, mientras que el backend generativo (SignJoey) queda preparado para entrenar y evaluar traducción secuencial con GPU.

Las limitaciones principales son el ruido de alineamiento de WhisperX y la ausencia de anotación explícita de señas/glosas. A futuro, el camino más directo es mejorar el dataset (alineación y segmentación), incorporar features multimodales (pose/cara) de forma consistente, y evaluar modelos generativos con criterios de latencia para acercarse a una experiencia más “tiempo real”.

## 6. Referencias

Listar referencias usadas en formato consistente. Como mínimo incluir:
- OpenCV documentation.
- MediaPipe Hands documentation.
- MobileNetV3 / transfer learning references.
- Paper de iLSU-T:
  Stassi, A., Boria, Y., Di Martino, M., & Randall, G. (2025). *iLSU-T: an Open Dataset for Uruguayan Sign Language Translation*.

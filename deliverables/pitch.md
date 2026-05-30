# Pitch (máx {{PITCH_MAX_MIN}} min) — {{TITLE}}

**Duración objetivo:** 7–{{PITCH_MAX_MIN}} min  

## Slide 1 — Título y propuesta

- **lsu-pria**: reconocimiento y traducción inicial de señas con visión computacional
- Proyecto final de **Visión Computacional / Procesamiento de Imágenes**
- Objetivo: detectar señas desde video y transformarlas en **texto + apoyo por voz**

## Slide 2 — Problema que queremos resolver

- La comunicación entre personas sordas usuarias de LSU y sistemas digitales todavía tiene muchas barreras
- En una interfaz convencional, la cámara “ve” una mano, pero no entiende el gesto ni su intención comunicativa
- Nuestro problema concreto: **capturar una seña en video y devolver una interpretación útil en tiempo real u offline**

## Slide 3 — ¿Qué es la LSU y por qué importa?

- La **Lengua de Señas Uruguaya (LSU)** es una lengua natural, visual y gestual usada por parte de la comunidad sorda en Uruguay
- No es “español en gestos”: tiene estructura propia, uso del espacio, movimiento, orientación, expresión facial y contexto
- Eso vuelve al problema interesante para visión computacional: no alcanza con reconocer una forma de mano aislada

## Slide 4 — Alcance realista del proyecto

- No buscamos resolver traducción libre completa de LSU en esta etapa
- Sí buscamos una **primera aproximación funcional** con:
  - reconocimiento de señas/palabras frecuentes,
  - análisis de video y webcam,
  - composición de texto,
  - y salida hablada mediante TTS
- El foco está en construir un pipeline defendible, reproducible y extensible

## Slide 5 — Aplicación que construimos

- **Web app** con React + FastAPI para demo y análisis visual
- **Demo desktop** con OpenCV para captura local
- Modos de uso:
  - webcam en vivo,
  - video local,
  - video de referencia / YouTube descargado
- La aplicación muestra overlays, predicción, logs, texto compuesto y puede reproducir voz

## Slide 6 — Cómo funciona técnicamente

- **OpenCV**: captura, preprocesado, máscara de piel, tracking y visualización
- **MediaPipe**: landmarks de mano, pose y componentes multimodales
- Pipelines principales:
  - A: landmarks → features geométricas → clasificador liviano
  - B: ROI de mano → CNN con transfer learning
  - C: secuencias multimodales → baseline temporal
- Pipeline adicional SLT offline para clips completos usando iLSU-T

## Slide 7 — Dataset y estrategia de entrenamiento

- Para la parte experimental de traducción usamos **iLSU-T**, dataset abierto de LSU orientado a traducción
- Como iLSU‑T no está etiquetado frame‑a‑frame, usamos WhisperX como supervisión por segmento:
  - segmentación (clip → texto WhisperX),
  - filtros (duración/longitud) para estabilidad,
  - splits por episodio (`group_id`) para evitar leakage,
  - export de features por clip,
  - entrenamiento + evaluación con métricas
- Baseline proxy (rápido) para demo reproducible + backend generativo (SignJoey) opcional en GPU (Colab)

## Slide 8 — Resultados y hallazgos

- El sistema ya permite:
  - reconocer señas/palabras en un conjunto acotado,
  - componer texto,
  - y reproducirlo en voz
- El enfoque **MediaPipe + OpenCV** resultó una base muy sólida para prototipado
- iLSU-T quedó integrado como camino serio hacia modelos de traducción (WhisperX → texto) y evaluación con métricas

{{SLT_PITCH_SECTION}}

## Slide 9 — Limitaciones y discusión

- La tarea sigue siendo difícil por:
  - desalineamiento texto↔señas (WhisperX),
  - variabilidad entre intérpretes y contextos,
  - y naturaleza secuencial del lenguaje
- Traducir LSU robustamente requiere más que mano: también importan pose, cara, movimiento y contexto
- Por eso planteamos el proyecto como una base escalable, no como solución cerrada

## Slide 10 — Cierre

- Aportamos una aplicación funcional y un flujo técnico reproducible
- El proyecto resuelve un problema real de accesibilidad desde visión computacional
- Dejamos preparado el camino para:
  - ampliar vocabulario,
  - mejorar modelos temporales,
  - y acercarnos progresivamente a traducción más robusta de LSU
- **Demo + preguntas**

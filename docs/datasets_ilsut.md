# iLSU-T (LSU) — revisión para entrenamiento de modelos del proyecto

Fuentes:
- Repo (código + metadata): `https://github.com/ariel-e-stassi/iLSU-T`
- Sitio de descarga (licencia + formulario): `https://iie.fing.edu.uy/proyectos/lsu-ds/ilsu-t/`

## Qué trae iLSU-T (según las fuentes)

- Dataset para **traducción** LSU↔texto con **videos RGB + audio + transcripciones**; el sitio reporta “more than 185 hours” de TV.  
- Descarga con **acceso controlado**: aceptar una licencia de uso restringido y completar un formulario; los links llegan por email.
- Para cada *source* (en el sitio): (1) *episodes*, (2) transcripciones WhisperX, (3) transcripciones WhisperX alineadas manualmente + *linguistic context labeling*.
- Estructura del repo (útil para reproducibilidad):
  - `preprocessing/`: métodos para obtener episodios desde material crudo + archivos de texto.
  - `data/`: CSV con episodios + metadata (puede requerir ajustar paths locales).
  - `video_clipping_and_visual_feats/`: notebook para clips y features (I3D).

## Qué tan usable es para *nuestros* modelos (clasificación de señas estáticas)

1) **No es un dataset “listo” para clasificación de gestos estáticos por clase**.
   - iLSU-T está pensado para *traducción* y organiza el material por episodios/segmentos alineados a texto; no está orientado a etiquetas “letra X / seña Y” por frame para alfabeto manual.

2) Sí es útil para:
   - **Pre-entrenamiento / extracción de features**: generar clips y/o features visuales (p.ej., I3D) para luego adaptar a tareas específicas.
   - **Mining de “palabras”** (señas tipo palabra) como *weak labels*: usar la segmentación temporal del texto (WhisperX) y, cuando exista, la alineación manual, para recortar clips candidatos.
   - **Benchmark cualitativo**: discutir explícitamente por qué el alineamiento texto↔seña es difícil (desfasajes, epéntesis, pausas), que impacta directo en “entrenar desde texto”.

## Recomendación práctica para el proyecto (tiempo/alcance)

- En esta entrega (sin dataset propio), usar iLSU‑T como **fuente principal**:
  - pipeline basado en segmentos WhisperX (clip → texto) para SLT, con métricas y un demo de “texto + voz”.
- La recolección de dataset propio por webcam (landmarks + ROI) queda como flujo opcional del repo, pero **no se usa** para esta demo/entrenamiento.

## Si deciden entrenar con iLSU-T: pipeline mínimo sugerido

1) Descargar iLSU-T desde el sitio (aceptar licencia, recibir links por email).
2) Usar WhisperX (y/o alineación manual si está disponible para ese source/episodio) para definir segmentos candidatos.
3) Recortar clips (ventanas temporales) y extraer ROI (mano/torso) + features (MediaPipe/OpenCV).
4) Etiquetado débil + filtrado:
   - filtrar por confianza, duración, presencia de manos, y remover casos ambiguos.
5) Entrenar un clasificador “palabra” (no letras) y reportar limitaciones/errores como parte de Resultados.

## Cómo mapea (o no) a nuestros 2 pipelines

- Pipeline A (MediaPipe landmarks + SVM): iLSU-T no trae landmarks listos; habría que correr MediaPipe sobre videos/clips para extraerlos y luego usar *pseudo-labels* desde texto.
- Pipeline B (ROI + CNN transfer learning): iLSU-T puede aportar **muchos frames/clips** para pre-entrenar un encoder sobre ROI (aunque sin labels directos de letra), o para un set chico *weakly supervised* a nivel “palabra/frasal”.

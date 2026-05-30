# AGENTS.md

Este archivo define instrucciones para agentes que trabajen en este repositorio.

## Alcance

Estas reglas aplican a todo el árbol del proyecto.

## Objetivo del repo

Proyecto final de Vision Computacional / Procesamiento de Imagenes para reconocimiento de senias en tiempo real.

El repo incluye:
- demo desktop con OpenCV,
- web app con FastAPI + React,
- pipelines de entrenamiento para landmarks, CNN y secuencias,
- flujo complementario para usar iLSU-T como fuente de weak labels.

## Reglas de trabajo

- Mantener cambios pequenos, enfocados y compatibles con el flujo existente.
- Preferir integraciones reutilizables sobre scripts aislados o hardcodeados.
- No asumir que iLSU-T esta descargado localmente; el dataset tiene acceso restringido.
- No agregar pasos de descarga automatica para iLSU-T.
- Mantener compatibilidad entre scripts CLI y scripts internos. Si una capacidad nueva existe en `scripts/*.py`, exponerla tambien en `lsupria.py` cuando tenga sentido.
- Evitar romper el flujo actual de:
  - `train_landmarks.py`
  - `train_cnn.py`
  - `eval_split.py`
  - `run_ablation.py`
  - `lsupria.py`

## Convenciones de datos

- Para datasets tabulares, preferir columnas explicitas y estables:
  - `label`
  - `subject_id`
  - `group_id`
  - `img_raw_path`
  - `img_masked_path`
- Para landmarks, mantener compatibilidad con ambos formatos:
  - columnas anchas `lm_*`
  - columna JSON `landmarks`
- Para splits agrupados, usar `group_id` cuando los datos provienen de iLSU-T o de otra fuente donde no conviene mezclar muestras del mismo episodio/fuente entre train y test.

## iLSU-T

- Tratar iLSU-T como dataset de traduccion, no como dataset nativo de letras frame-a-frame.
- El flujo esperado es:
  1. `ilsut_make_manifest.py`
  2. `ilsut_extract_rois.py`
  3. `train_landmarks.py` y/o `train_cnn.py`
  4. `eval_split.py`
- Mantener autodeteccion de columnas y resolucion robusta de paths del `episodes.csv`.
- No asumir nombres exactos de columnas si pueden inferirse.

## Validacion

- Antes de cerrar cambios en scripts Python, correr al menos:
  - `python3 -m py_compile ...` sobre los archivos tocados
- Si se tocan argumentos CLI, validar `--help` del comando afectado.
- Si se cambian formatos CSV, revisar compatibilidad con entrenamiento y evaluacion.

## Documentacion

- Si se agrega un comando nuevo de uso general, actualizar:
  - `README.md`
  - documentacion especifica en `docs/` si aplica


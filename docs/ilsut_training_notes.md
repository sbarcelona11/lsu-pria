# iLSU-T — guía rápida para entrenar (weak labels) en este repo

Objetivo: usar iLSU-T como fuente *complementaria* para experimentar con clips/ROI en señas tipo palabra (p.ej. “hola”, “gracias”), sabiendo que iLSU-T es un dataset de **traducción** (video+audio+texto) y no de “letras” frame-a-frame.

## 1) Descargar y extraer iLSU-T

- La descarga se gestiona desde el sitio del dataset (licencia + formulario).  
- Extraer los `.rar` y dejar un `ROOT` local con los episodios y WhisperX.

En este repo ya quedaron cargados los links de `source2` y `source3` en:
- `deliverables/ilsut_downloads.json`

Descarga desde el CLI:

```bash
python3 vcpria.py ilsut-download \
  --out-dir data/ilsut_archives \
  --sources source2 source3 \
  --skip-existing
```

Eso deja los `.rar` organizados por fuente dentro de `data/ilsut_archives/`.

Extraccion desde el CLI:

```bash
python3 vcpria.py ilsut-extract \
  --archives-dir data/ilsut_archives \
  --out-root data/ilsut_extracted \
  --sources source2 source3 \
  --skip-existing
```

Eso deja una estructura como:
- `data/ilsut_extracted/source2/episodes/...`
- `data/ilsut_extracted/source2/whisperx/...`
- `data/ilsut_extracted/source3/episodes/...`
- `data/ilsut_extracted/source3/whisperx/...`

## 2) Construir un manifest de segmentos (weak labels)

El manifest es un CSV con filas:
`video_path,start_ms,end_ms,label,matched_word,...`

1) Crear un JSON de keywords (label -> lista de patrones). Ejemplo:
- `deliverables/ilsut_keywords.example.json`

2) Ejecutar:

```bash
python3 scripts/ilsut_make_manifest.py \
  --episodes-csv /ABS/PATH/a/episodes.csv \
  --root /ABS/PATH/al/ROOT_ILSUT \
  --keywords deliverables/ilsut_keywords.example.json \
  --out data/ilsut_manifest.csv
```

Notas:
- `--episodes-csv` es el CSV de episodios/metadata (viene en el repo oficial iLSU-T, carpeta `data/`).
- El script ahora intenta autodetectar columnas como `episode_id`, `video_path` y `whisperx_path`, y además resuelve rutas “rotas” buscando por nombre de archivo dentro de `--root`.

## 3) Extraer ROIs (y opcional landmarks) desde el manifest

```bash
python3 scripts/ilsut_extract_rois.py \
  --manifest data/ilsut_manifest.csv \
  --out-dir data/ilsut_weak_roi \
  --fps 5 \
  --max-per-seg 30 \
  --preprocess \
  --skin-mask --save-masked \
  --save-landmarks
```

Esto genera:
- `data/ilsut_weak_roi/images_raw/<label>/*.jpg`
- `data/ilsut_weak_roi/images_masked/<label>/*.jpg` (si se habilita)
- `data/ilsut_weak_roi/landmarks.csv` (si se habilita)

## 4) Entrenar modelos con ese output

- Pipeline B (CNN): entrenar desde `samples.csv` usando split agrupado:

```bash
python3 vcpria.py train-cnn \
  --csv data/ilsut_weak_roi/samples.csv \
  --image-col img_raw_path \
  --group-col group_id \
  --out models/cnn_ilsut.pt
```

- Pipeline A (landmarks): entrenar desde `landmarks.csv`:

```bash
python3 vcpria.py train-landmarks \
  --csv data/ilsut_weak_roi/landmarks.csv \
  --out models/landmarks_ilsut.joblib
```

Recomendación: reportar esto como “weak supervision” (texto -> clips) y discutir errores esperables (desfasaje, epéntesis, etc.).

## 5) Flujo integrado

Si querés hacer preparación + entrenamiento en un solo paso:

```bash
python3 vcpria.py ilsut-train \
  --episodes-csv /ABS/PATH/episodes.csv \
  --root /ABS/PATH/iLSUT_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir data/ilsut_run \
  --pipelines cnn landmarks \
  --preprocess \
  --skin-mask \
  --camera-like \
  --cnn-epochs 10
```

Este comando:
- arma `manifest.csv` desde WhisperX,
- extrae ROI y genera `samples.csv` + `landmarks.csv`,
- entrena `cnn_ilsut.pt` y/o `landmarks_ilsut.joblib`,
- usa `group_id` como split agrupado por defecto para evitar mezclar episodios/fuentes del mismo bloque.

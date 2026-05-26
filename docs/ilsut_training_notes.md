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

Conversión opcional de videos `.avi` a `.mp4`:

```bash
python3 vcpria.py ilsut-convert-videos \
  --root data/ilsut_extracted \
  --sources source2 source3 \
  --output-ext .mp4 \
  --skip-existing
```

Notas:
- Si existe `ffmpeg`, el script lo usa automáticamente.
- Si no existe `ffmpeg`, hace fallback con OpenCV.
- La conversión mantiene la estructura por fuente y carpeta.

## 2) Construir un manifest de segmentos (weak labels)

El manifest es un CSV con filas:
`video_path,start_ms,end_ms,label,matched_word,...`

Si no tenés el `episodes.csv` oficial, podés generarlo directo desde los archivos extraídos:

```bash
python3 vcpria.py ilsut-build-episodes-csv \
  --root data/ilsut_extracted \
  --sources source2 \
  --out data/ilsut_source2_episodes_generated.csv
```

Si querés medir qué clases tienen soporte real antes de entrenar, podés correr:

```bash
python3 vcpria.py ilsut-analyze-support \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_support_s2s3 \
  --sources source2 source3 \
  --min-label-count 20
```

Eso te deja:
- `runs/ilsut_support_s2s3/manifest.csv`
- `runs/ilsut_support_s2s3/label_support.csv`
- `runs/ilsut_support_s2s3/recommended_labels.json`
- `runs/ilsut_support_s2s3/label_support.md`

La idea es usarlo para decidir el vocabulario entrenable real con `source2 + source3` antes de correr `ilsut-train` o `ilsut-train-mm`.

Si querés revisar qué transcriptos está matcheando cada regla y afinar el JSON con evidencia:

```bash
python3 vcpria.py ilsut-audit-keywords \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_keywords_audit_s2 \
  --sources source2 \
  --manifest-limit 12
```

Eso te deja:
- `runs/ilsut_keywords_audit_s2/keywords_audit.json`
- `runs/ilsut_keywords_audit_s2/keywords_audit.md`

Además del ejemplo amplio, el repo ahora incluye un vocabulario más enfocado para una primera corrida robusta:
- `deliverables/ilsut_keywords.focused.json`

Si querés acercarte más al flujo oficial orientado a traducción por clips, podés preparar un subset con splits reproducibles:

```bash
python3 vcpria.py ilsut-prepare-slt-subset \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_slt_subset_s2s3 \
  --sources source2 source3 \
  --min-label-count 20 \
  --export-clips
```

Eso te deja:
- `runs/ilsut_slt_subset_s2s3/subset_manifest.csv`
- `runs/ilsut_slt_subset_s2s3/train.csv`
- `runs/ilsut_slt_subset_s2s3/val.csv`
- `runs/ilsut_slt_subset_s2s3/test.csv`
- `runs/ilsut_slt_subset_s2s3/subset_info.json`
- `runs/ilsut_slt_subset_s2s3/clips/` y `clips_index.csv` si activás `--export-clips`

`labels-json` puede apuntar al `recommended_labels.json` generado por `ilsut-analyze-support` para congelar exactamente el vocabulario elegido.

Ese CSV queda con columnas como:
- `episode_id`
- `source`
- `video_path`
- `whisperx_path`

Si en `episodes/` conviven `.avi` y `.mp4` para el mismo episodio, el generador prioriza `.mp4` y deja una sola fila por episodio.

1) Crear un JSON de keywords. Ejemplo:
- `deliverables/ilsut_keywords.example.json`

Formatos soportados:

```json
{
  "si": ["si", "\\\\bs[ií]\\\\b"]
}
```

o formato extendido:

```json
{
  "gracias": {
    "include": ["gracias", "agradezco"],
    "phrases": ["muchas gracias", "te agradezco"],
    "exclude": ["gracia"],
    "phrase_exclude": ["gracias a dios"]
  }
}
```

Claves útiles:
- `include`: palabras o expresiones simples
- `variants`: alias de `include`
- `regex`: regex por palabra/expresión
- `fuzzy`: matching aproximado por palabra usando `rapidfuzz`
- `phrases`: frases multi-palabra
- `phrase_regex`: regex de frase
- `fuzzy_phrases`: matching aproximado de frases usando `rapidfuzz`
- `exclude`: exclusiones sobre match simple
- `exclude_regex`: exclusiones regex simples
- `phrase_exclude`: exclusiones de frases
- `phrase_exclude_regex`: exclusiones regex de frases

Notas:
- `fuzzy` y `fuzzy_phrases` son opcionales y útiles cuando WhisperX mete errores leves de escritura.
- Si `rapidfuzz` no está instalado, esas reglas simplemente no hacen match.

2) Ejecutar:

```bash
python3 scripts/ilsut_make_manifest.py \
  --episodes-csv /ABS/PATH/a/episodes.csv \
  --root /ABS/PATH/al/ROOT_ILSUT \
  --keywords deliverables/ilsut_keywords.example.json \
  --out data/ilsut_manifest.csv
```

Notas:
- `--episodes-csv` puede ser el CSV oficial de episodios/metadata (repo iLSU-T, carpeta `data/`) o uno generado con `ilsut-build-episodes-csv`.
- El script ahora intenta autodetectar columnas como `episode_id`, `video_path` y `whisperx_path`, y además resuelve rutas “rotas” buscando por nombre de archivo dentro de `--root`.
- Si el CSV oficial referencia `.avi` pero en tu árbol local ya existe `.mp4` o `.mkv`, el manifest usa automáticamente la variante convertida sin necesidad de editar el CSV.
- En una estructura como `data/ilsut_extracted/source2/{episodes,whisperx}`, el flujo actual ya funciona sin reorganizar carpetas.

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
  --root /ABS/PATH/iLSUT_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir data/ilsut_run \
  --sources source2 \
  --pipelines cnn landmarks \
  --min-label-count 20 \
  --preprocess \
  --skin-mask \
  --camera-like \
  --cnn-epochs 10
```

Este comando:
- si no le pasás `--episodes-csv`, genera `episodes_generated.csv` dentro del `work-dir`,
- arma `manifest.csv` desde WhisperX,
- opcionalmente filtra clases flojas con `--min-label-count`,
- extrae ROI y genera `samples.csv` + `landmarks.csv`,
- entrena `cnn_ilsut.pt` y/o `landmarks_ilsut.joblib`,
- genera `eval_split.json`, `dataset_stats.md` y, si no se desactiva, `ablation_table.md` + `report.md`,
- usa `group_id` como split agrupado por defecto para evitar mezclar episodios/fuentes del mismo bloque.

## 6) Preset práctico para `source2`

Si querés iterar rápido sobre la estructura real de `data/ilsut_extracted/source2`, podés usar:

```bash
python3 vcpria.py ilsut-preset \
  --root data/ilsut_extracted \
  --source source2 \
  --mode both \
  --preset quick
```

Esto genera dos workspaces:
- `runs/ilsut_presets/source2_quick_frame`
- `runs/ilsut_presets/source2_quick_multimodal`

Preset `quick`:
- frame: `manifest-limit=12`, `min-label-count=20`, `fps=2`, `max-per-seg=12`, `cnn-epochs=3`, `--skip-ablation`
- multimodal: `manifest-limit=12`, `min-label-count=20`, `fps=3`, `max-per-seg=16`, `window=20`, `min-frames=6`

Si ya querés una corrida más seria:

```bash
python3 vcpria.py ilsut-preset \
  --root data/ilsut_extracted \
  --source source2 \
  --mode both \
  --preset standard \
  --cnn-device mps
```

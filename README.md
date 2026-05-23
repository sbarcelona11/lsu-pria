# VC-pria — Reconocimiento de señas en tiempo real

Incluye dos pipelines:

- **A (baseline)**: MediaPipe Hands → landmarks → features geométricas → clasificador (SVM/RF).
- **B (transfer learning)**: recorte ROI de mano (OpenCV) → CNN (MobileNetV3) fine-tuned.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)

pip install -r requirements.txt
```

## CLI unificado (recomendado)

```bash
python3 vcpria.py --help
python3 vcpria.py web --landmarks-model models/landmarks.joblib --cnn-model models/cnn.pt --open-browser
```

## Web app (frase + TTS)

Levanta una web local (abre en el navegador) que:
- captura webcam en el browser,
- envía frames al backend para inferencia,
- confirma tokens por estabilidad + pausa,
- reproduce la frase con TTS offline (pyttsx3).

```bash
python3 scripts/run_webapp.py --landmarks-model models/landmarks.joblib --cnn-model models/cnn.pt --host 127.0.0.1 --port 8000 --open-browser
```

Abrir: `http://127.0.0.1:8000`

Nota: se usa `mediapipe==0.10.18` porque esta API requiere `mp.solutions.hands`.

Tip: la UI muestra arriba el estado de modelos cargados y el conteo de clases por pipeline.

### UI “pulida” (React)

La UI React vive en `web-ui/`.

- Dev (API + React dev server):
  - `bash scripts/run_web_ui_dev.sh`
  - Abrir `http://127.0.0.1:5173`
- Build (para servir desde FastAPI en `/`):
  - `bash scripts/build_web_ui.sh`
  - Luego correr el servidor FastAPI y va a servir `web-ui/dist` automáticamente.

Notas:
- En modo dev, la UI usa proxy (same-origin) por defecto. Si querés apuntar a otra API, setear `VITE_API_BASE` en `web-ui/.env.development`.
- La UI legacy (HTML inline) queda disponible en `http://127.0.0.1:8000/legacy`.

Controles en la web:
- `Compose mode`: `both | words | spelling`
- `tracker (bbox fallback)`: si MediaPipe pierde la mano, mantiene un bbox para que el pipeline CNN siga funcionando.
- `draw landmarks/bbox`: dibuja bbox y puntos/conexiones sobre el video (debug)
- `mask thumbnail` / `ROI thumbnail`: miniaturas de debug (útiles para discutir segmentación/ROI)
- `compose debug`: muestra contadores internos (estabilidad/candidato) para ajustar parámetros
- Botones: `Espacio`, `Borrar`, `Reset frase`, `Hablar (TTS)`

Export de sesión (texto + tokens + log de inferencias):
- `GET /api/session/{session_id}/export`
- `GET /api/session/{session_id}/export.csv`

## Entrega (PDF + pitch + zip)

Plantillas en `deliverables/` y generación:

```bash
python3 scripts/build_deliverables.py
python3 scripts/package_submission.py
python3 scripts/check_deliverables.py
```

One-shot:
```bash
python3 scripts/prepare_moodle_submission.py
# o usando el CLI:
python3 vcpria.py deliverables
```

Si querés setear config desde CLI (sin editar JSON):
```bash
python3 vcpria.py deliverables --set-group "Grupo 7" --set-members "Nombre1" "Nombre2" --set-date 2026-05-28
```

Salida:
- `deliverables/out/informe.pdf`
- `deliverables/out/pitch.pdf`
- `deliverables/out/pitch.pptx`
- `deliverables/out/entrega_moodle.zip`

## Temporal (señas dinámicas) — baseline

Recolectar secuencias (1 sample por tecla, por ~2s):
```bash
python3 vcpria.py collect-seq --out data/seq_S1 --subject-id S1 --labels hola gracias no
```

Entrenar:
```bash
python3 vcpria.py train-seq --seq-dir data/seq_S1 --out models/sequence.joblib
```

Demo desktop:
```bash
python3 vcpria.py demo --pipeline sequence --model models/sequence.joblib
```

## iLSU-T (complementario, weak labels)

Ver guía: `docs/ilsut_training_notes.md`

Pipeline recomendado de punta a punta:
- `docs/end_to_end_pipeline.md`

Flujo integrado para dataset propio:

```bash
python3 vcpria.py train-stack \
  --csvs data/S1/landmarks.csv data/S2/landmarks.csv \
  --work-dir runs/demo_stack \
  --cnn-image-col img_raw_path \
  --group-col subject_id
```

Flujo integrado para preparar y entrenar desde iLSU-T:

```bash
python3 vcpria.py ilsut-download \
  --out-dir data/ilsut_archives \
  --sources source2 source3 \
  --skip-existing

python3 vcpria.py ilsut-extract \
  --archives-dir data/ilsut_archives \
  --out-root data/ilsut_extracted \
  --sources source2 source3 \
  --skip-existing

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

Salida esperada:
- `data/ilsut_run/manifest.csv`
- `data/ilsut_run/prepared/samples.csv`
- `data/ilsut_run/prepared/landmarks.csv`
- `data/ilsut_run/models/cnn_ilsut.pt`
- `data/ilsut_run/models/landmarks_ilsut.joblib`

## Recolección de datos (webcam)

1) Crear dataset (landmarks + ROI imagen). Presionar teclas para etiquetar; `Esc` para salir:

```bash
python3 scripts/collect_data.py --out data/collected --labels A B C hola gracias si no ayuda
```

Tips multi-sujeto (para split por persona):
```bash
python3 vcpria.py collect --out data/S1 --subject-id S1 --labels A B C hola gracias si no ayuda
python3 vcpria.py collect --out data/S2 --subject-id S2 --labels A B C hola gracias si no ayuda
python3 vcpria.py merge-csvs --inputs data/S1/landmarks.csv data/S2/landmarks.csv --out data/merged/landmarks.csv
python3 vcpria.py eval-split --csv data/merged/landmarks.csv --landmarks-model models/landmarks.joblib --group-col subject_id
```

Stats del dataset (conteos por clase y sujeto):
```bash
python3 vcpria.py dataset-stats --csv data/merged/landmarks.csv --by label_subject --out-md results/dataset_stats.md
```

Validación multi-sujeto (mínimo de muestras por clase y sujeto):
```bash
python3 vcpria.py validate-multisubject --csv data/merged/landmarks.csv --min-per-label-per-subject 30 --out-md results/multisubject_check.md
```

Esto guarda:
- `data/collected/landmarks.csv`
- `data/collected/images_raw/<label>/*.jpg`
- `data/collected/images_masked/<label>/*.jpg`

## Entrenamiento

### Pipeline A (landmarks)

```bash
python scripts/train_landmarks.py --csv data/collected/landmarks.csv --out models/landmarks.joblib
```

### Pipeline B (CNN transfer learning)

```bash
python3 scripts/train_cnn.py --img-dir data/collected/images_masked --out models/cnn.pt --epochs 10
```

## Demo en tiempo real

```bash
python3 main.py --pipeline landmarks --model models/landmarks.joblib
# o:
python3 main.py --pipeline cnn --model models/cnn.pt
```

Teclas en demo:
- `q` salir
- `m` toggle máscara piel (debug)
- `k` alternar máscara `YCrCb`/`HSV`
- `t` toggle tracker
- `p` toggle preprocesado (CLAHE/denoise)

## Evaluación rápida (report + confusiones)

```bash
python3 scripts/eval_models.py --csv data/collected/landmarks.csv --landmarks-model models/landmarks.joblib --cnn-model models/cnn.pt
```

## Evaluación con split (más realista)

```bash
python3 scripts/eval_split.py --csv data/collected/landmarks.csv --landmarks-model models/landmarks.joblib --test-size 0.2
python3 scripts/eval_split.py --csv data/collected/landmarks.csv --cnn-model models/cnn.pt --cnn-image-col img_masked_path --test-size 0.2
```

## Perfilado de FPS (webcam)

```bash
python3 scripts/profile_fps.py --pipeline landmarks --model models/landmarks.joblib --seconds 10
python3 scripts/profile_fps.py --pipeline cnn --model models/cnn.pt --seconds 10
# ejemplos: sin tracker, y con máscara HSV
python3 scripts/profile_fps.py --pipeline cnn --model models/cnn.pt --seconds 10 --no-tracker
python3 scripts/profile_fps.py --pipeline cnn --model models/cnn.pt --seconds 10 --skin-mask --mask-space hsv
```

## Export de sesión web a JSON

Con la web corriendo, el `session_id` se muestra arriba (y hay botón `Export JSON`). Alternativamente, exportar por script:

```bash
python3 scripts/export_session.py --base-url http://127.0.0.1:8000 --session-id <SESSION_ID> --out exports/session.json
```

Export de log a CSV:
```bash
python3 scripts/export_session_csv.py --base-url http://127.0.0.1:8000 --session-id <SESSION_ID> --out exports/session_log.csv
```

## Tabla “precisión vs FPS” (para informe)

Después de correr `eval_split.py` (macro-F1) y `profile_fps.py` (FPS), volcar manualmente los números:

```bash
python3 scripts/make_results_table.py --out results/table.md --rows \
  landmarks 0.82 24.5 \
  cnn_masked 0.88 14.2
```

## Ablation automático (genera tabla)

Corre split-eval (macro-F1) + FPS profiling (webcam) y genera:
- `results/ablation_table.md`
- `results/ablation_results.json`

```bash
python3 vcpria.py ablation \
  --csv data/collected/landmarks.csv \
  --landmarks-model models/landmarks.joblib \
  --cnn-model models/cnn.pt \
  --seconds 10
```

## Ablation grid (sweep de toggles OpenCV)

Hace sweep de `preprocess/tracker/skin_mask/mask_space` y genera tabla con Macro-F1 (split) + FPS:

```bash
python3 vcpria.py ablation-grid \
  --csv data/merged/landmarks.csv \
  --landmarks-model models/landmarks.joblib \
  --cnn-model models/cnn.pt \
  --group-col subject_id \
  --seconds 10 \
  --out-md results/ablation_grid.md \
  --out-json results/ablation_grid.json
```

## Reporte rápido (Markdown + plot)

Genera `results/report.md` y un scatter plot `results/precision_vs_fps.png` (requiere `run_ablation.py` y opcionalmente `dataset_stats.py`):

```bash
python3 scripts/make_report.py
```

## Notas
- Para discusión: medir **FPS** y comparar precisión vs latencia entre pipelines.
- MediaPipe + landmarks suele ser más rápido/estable con pocos datos; CNN puede mejorar en clases similares si el dataset es suficiente.

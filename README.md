# VC-pria — Reconocimiento de señas en tiempo real

Incluye varios pipelines:

- **A (baseline)**: MediaPipe Hands → landmarks → features geométricas → clasificador (SVM/RF).
- **B (transfer learning)**: recorte ROI de mano (OpenCV) → CNN (MobileNetV3) fine-tuned.
- **C (temporal)**: secuencias de landmarks → baseline temporal.
- **D (multimodal)**: manos + pose + cara → baseline temporal multimodal.

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
python3 vcpria.py web --landmarks-model models/landmarks.joblib --cnn-model models/cnn.pt --multimodal-model models/multimodal_sequence.joblib --open-browser
```

## Web app (frase + TTS)

Levanta una web local (abre en el navegador) que:
- captura webcam en el browser,
- envía frames al backend para inferencia,
- confirma tokens por estabilidad + pausa,
- reproduce la frase con TTS offline (pyttsx3).

```bash
python3 scripts/run_webapp.py --landmarks-model models/landmarks.joblib --cnn-model models/cnn.pt --multimodal-model models/multimodal_sequence.joblib --host 127.0.0.1 --port 8000 --open-browser
```

Abrir: `http://127.0.0.1:8000`

Nota: se usa `mediapipe==0.10.18` porque esta API requiere `mp.solutions.hands`.

Tip: la UI muestra arriba el estado de modelos cargados y el conteo de clases por pipeline.
Tambien puede mostrar `multimodal` si cargan `multimodal_sequence.joblib`.

Desarrollo backend + frontend juntos:

```bash
python3 vcpria.py app-dev --install-ui
```

Si ya entrenaste con `train-stack`:

```bash
python3 vcpria.py app-dev --stack-work-dir runs/demo_stack
```

Si el `work-dir` tambien tiene `models/multimodal_sequence.joblib`, la web lo detecta automaticamente.

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
- `multimodal`: usa manos + pose + cara para el baseline temporal robusto.
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

## LSU robusto — baseline multimodal

Para ir más allá de mano aislada, el repo ahora incluye un baseline secuencial con:
- mano izquierda,
- mano derecha,
- pose superior,
- cara reducida.

Recolección webcam:
```bash
python3 vcpria.py collect-mm-seq --out data/mm_seq_S1 --subject-id S1 --labels hola gracias no si ayuda
```

Entrenamiento:
```bash
python3 vcpria.py train-mm-seq --seq-dir data/mm_seq_S1 --out models/mm_sequence.joblib
```

Flujo local integrado:
```bash
python3 vcpria.py train-mm-stack \
  --seq-dirs data/mm_seq_S1 data/mm_seq_S2 \
  --work-dir runs/mm_stack \
  --group-col subject_id
```

Entrenamiento directamente desde videos etiquetados:
```bash
python3 vcpria.py train-videos-mm-stack \
  --videos-root data/train_videos \
  --work-dir runs/mm_from_videos \
  --layout label \
  --preprocess
```

Entrenamiento desde iLSU-T weak labels:
```bash
python3 vcpria.py ilsut-train-mm \
  --root /ABS/PATH/iLSUT_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_mm \
  --sources source2 \
  --preprocess
```

Este baseline no resuelve traducción completa de LSU, pero sí deja al proyecto parado sobre una arquitectura más realista para:
- señas dinámicas,
- contexto temporal,
- información no manual.

## iLSU-T (complementario, weak labels)

Ver guía: `docs/ilsut_training_notes.md`

Pipeline recomendado de punta a punta:
- `docs/end_to_end_pipeline.md`
- `docs/final_demo_runbook.md`
- `docs/video_training.md`

Flujo integrado para dataset propio:

```bash
python3 vcpria.py train-stack \
  --csvs data/S1/landmarks.csv data/S2/landmarks.csv \
  --work-dir runs/demo_stack \
  --cnn-image-col img_raw_path \
  --group-col subject_id

python3 vcpria.py demo-stack-web \
  --work-dir runs/demo_stack \
  --open-browser
```

Entrenamiento directo desde videos etiquetados:
```bash
python3 vcpria.py train-videos-stack \
  --videos-root data/train_videos \
  --work-dir runs/demo_videos \
  --layout label \
  --preprocess \
  --skin-mask \
  --cnn-device mps
```

Layouts soportados para `--videos-root`:
- `--layout label`: `data/train_videos/<label>/clip1.mp4`
- `--layout subject_label`: `data/train_videos/<subject>/<label>/clip1.mp4`
- `--layout auto`: detecta `subject/label/video` si hay dos niveles; si no, usa `label/video`

En estos comandos cada video queda como `group_id` propio, para evitar mezclar frames del mismo clip entre train y test.

Validacion con videos externos:

```bash
python3 vcpria.py validate-videos \
  --pipeline cnn \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/demo_stack/video_validation \
  --mode both \
  --preprocess \
  --skin-mask \
  --use-tracker
```

O para el baseline multimodal:

```bash
python3 vcpria.py validate-videos \
  --pipeline multimodal \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/mm_stack/video_validation \
  --mode both \
  --preprocess \
  --use-tracker
```

Comparacion automatica entre pipelines sobre los mismos videos:

```bash
python3 vcpria.py compare-video-pipelines \
  --pipelines cnn multimodal \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/model_compare \
  --mode both \
  --preprocess \
  --use-tracker
```

Salida:
- `runs/model_compare/compare_video_pipelines.json`
- `runs/model_compare/compare_video_pipelines.md`

Y para dejar sugerido el pipeline final de demo:

```bash
python3 vcpria.py recommend-demo-pipeline \
  --compare-json runs/model_compare/compare_video_pipelines.json \
  --out-dir runs/model_compare/recommendation \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json
```

Salida:
- `runs/model_compare/recommendation/recommended_demo_pipeline.json`
- `runs/model_compare/recommendation/recommended_demo_pipeline.md`

One-shot para cerrar la seleccion final:

```bash
python3 vcpria.py finalize-demo-selection \
  --out-dir runs/final_demo_selection \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --frame-work-dir runs/demo_stack \
  --multimodal-work-dir runs/mm_stack \
  --preprocess \
  --use-tracker
```

Eso genera:
- `runs/final_demo_selection/compare/compare_video_pipelines.json`
- `runs/final_demo_selection/compare/compare_video_pipelines.md`
- `runs/final_demo_selection/recommendation/recommended_demo_pipeline.json`
- `runs/final_demo_selection/recommendation/recommended_demo_pipeline.md`
- `runs/model_compare/<pipeline>/video_validation.json`

En la web React también hay una sección **Video analysis**:
- pegás una URL de YouTube/directa o subís un archivo,
- el backend procesa el video con OpenCV + MediaPipe + el pipeline elegido,
- genera un MP4 con overlay y devuelve texto + TTS.

Notas:
- Para URLs de YouTube se usa `yt-dlp`.
- Los videos procesados quedan expuestos en `/artifacts/<job_id>/processed.mp4`.

Notebook para Google Colab:
- `/Users/sebastian/Documents/VC-pria/notebooks/vc_pria_colab_training.ipynb`
- pensado para correr `train-stack`, `train-videos-stack` o `ilsut-train` con GPU de Colab para la CNN.
- `/Users/sebastian/Documents/VC-pria/notebooks/vc_pria_colab_multimodal_training.ipynb`
- pensado para correr `train-mm-stack`, `train-videos-mm-stack` o `ilsut-train-mm` en Colab.
- `/Users/sebastian/Documents/VC-pria/notebooks/vc_pria_colab_video_training.ipynb`
- pensado específicamente para entrenar directo desde carpetas de videos en Drive.

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

python3 vcpria.py ilsut-convert-videos \
  --root data/ilsut_extracted \
  --sources source2 source3 \
  --output-ext .mp4 \
  --skip-existing

python3 vcpria.py ilsut-build-episodes-csv \
  --root data/ilsut_extracted \
  --sources source2 \
  --out data/ilsut_source2_episodes_generated.csv

python3 vcpria.py ilsut-analyze-support \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_support_s2s3 \
  --sources source2 source3 \
  --min-label-count 20

python3 vcpria.py ilsut-audit-keywords \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_keywords_audit_s2 \
  --sources source2 \
  --manifest-limit 12

python3 vcpria.py ilsut-prepare-slt-subset \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir runs/ilsut_slt_subset_s2s3 \
  --sources source2 source3 \
  --min-label-count 20 \
  --export-clips

# si episodes/ tiene .avi y .mp4 para el mismo episodio, prioriza .mp4

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

Preset recomendado para `source2` sobre `data/ilsut_extracted`:

```bash
python3 vcpria.py ilsut-preset \
  --root data/ilsut_extracted \
  --source source2 \
  --mode both \
  --preset quick
```

El preset `quick` limita el manifest a `12` episodios y filtra clases con menos de `20` segmentos weak-labeled para iterar más rápido sin arrastrar clases casi vacías.

Si querés decidir primero qué vocabulario conviene entrenar con `source2 + source3`, el comando `ilsut-analyze-support` te deja:
- `manifest.csv`
- `label_support.csv`
- `recommended_labels.json`
- `label_support.md`

Si querés entender mejor qué transcriptos están disparando cada regla, `ilsut-audit-keywords` te deja:
- `keywords_audit.json`
- `keywords_audit.md`

Para una primera corrida más robusta, también tenés un vocabulario enfocado en las clases más fuertes detectadas hasta ahora:
- `/Users/sebastian/Documents/VC-pria/deliverables/ilsut_keywords.focused.json`

Si querés un flujo más cercano al uso oficial del dataset para traducción por clips, `ilsut-prepare-slt-subset` te deja:
- `subset_manifest.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `subset_info.json`
- `clips/` y `clips_index.csv` si activás `--export-clips`

Salida esperada:
- `data/ilsut_run/manifest.csv`
- `data/ilsut_run/prepared/samples.csv`
- `data/ilsut_run/prepared/landmarks.csv`
- `data/ilsut_run/models/cnn_ilsut.pt`
- `data/ilsut_run/models/landmarks_ilsut.joblib`
- `data/ilsut_run/results/eval_split.json`
- `data/ilsut_run/results/dataset_stats.md`
- `data/ilsut_run/results/ablation_table.md`
- `data/ilsut_run/results/report.md`

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

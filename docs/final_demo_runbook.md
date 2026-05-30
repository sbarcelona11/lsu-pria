# Runbook de demo final

Objetivo: tener una secuencia corta y repetible para el dia de la presentacion.

## 1. Entrenar el stack local

```bash
python3 lsupria.py train-stack \
  --csvs data/S1/landmarks.csv data/S2/landmarks.csv \
  --work-dir runs/demo_stack \
  --cnn-image-col img_raw_path \
  --group-col subject_id
```

Salida importante:
- `runs/demo_stack/models/landmarks.joblib`
- `runs/demo_stack/models/cnn.pt`
- `runs/demo_stack/results/eval_split.json`
- `runs/demo_stack/results/ablation_table.md`
- `runs/demo_stack/results/precision_vs_fps.png`

## 2. Levantar la demo web con esos modelos

```bash
python3 lsupria.py demo-stack-web \
  --work-dir runs/demo_stack \
  --open-browser
```

Si `runs/demo_stack/models/multimodal_sequence.joblib` existe, la web habilita tambien el pipeline `multimodal`.
Si levantan la web con `--slt-model`, la seccion de analisis de video tambien habilita `slt` para traduccion offline.

## 3. Mostrar durante el pitch

- La webcam en vivo.
- El overlay de landmarks y bbox.
- El cambio entre `landmarks`, `cnn` y `multimodal` si lo tienen entrenado.
- Si quieren mostrar traduccion por clip completo, usar la seccion de video con `slt`.
- La composicion de frase.
- Export de CSV/JSON si quieren mostrar trazabilidad.

## 4. Graficos y resultados para slides

Usar:
- `runs/demo_stack/results/ablation_table.md`
- `runs/demo_stack/results/precision_vs_fps.png`
- `runs/demo_stack/results/dataset_stats.md`

## 4.b Validacion con videos externos

Si descargan videos de referencia y quieren medir si la app compone bien la frase:

```bash
python3 lsupria.py validate-videos \
  --pipeline cnn \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/demo_stack/video_validation \
  --mode both \
  --preprocess \
  --skin-mask \
  --use-tracker
```

Salida:
- `runs/demo_stack/video_validation/video_validation.json`
- `runs/demo_stack/video_validation/video_validation.csv`

Si quieren validar el baseline multimodal:

```bash
python3 lsupria.py validate-videos \
  --pipeline multimodal \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/mm_stack/video_validation \
  --mode both \
  --preprocess \
  --use-tracker
```

Y para decidir cuál mostrar en el pitch:

```bash
python3 lsupria.py compare-video-pipelines \
  --pipelines cnn multimodal slt \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --slt-model runs/ilsut_slt_train_s2s3/models/slt_proxy.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --out-dir runs/model_compare \
  --mode both \
  --preprocess \
  --use-tracker
```

Y para convertir esa comparación en una recomendación operativa:

```bash
python3 lsupria.py recommend-demo-pipeline \
  --compare-json runs/model_compare/compare_video_pipelines.json \
  --out-dir runs/model_compare/recommendation \
  --cnn-model runs/demo_stack/models/cnn.pt \
  --multimodal-model runs/mm_stack/models/multimodal_sequence.joblib \
  --slt-model runs/ilsut_slt_train_s2s3/models/slt_proxy.joblib \
  --cases-json deliverables/youtube_validation_cases.example.json
```

Si ya tienen workspaces entrenados y quieren resolver todo de una:

```bash
python3 lsupria.py finalize-demo-selection \
  --out-dir runs/final_demo_selection \
  --cases-json deliverables/youtube_validation_cases.example.json \
  --frame-work-dir runs/demo_stack \
  --multimodal-work-dir runs/mm_stack \
  --slt-work-dir runs/ilsut_slt_train_s2s3 \
  --preprocess \
  --use-tracker
```

Y si el entrenamiento final sale directamente desde videos etiquetados:

```bash
python3 lsupria.py train-videos-mm-stack \
  --videos-root data/train_videos \
  --work-dir runs/mm_from_videos \
  --layout label \
  --preprocess
```

## 5. Entrega

Regenerar entregables finales:

```bash
python3 lsupria.py deliverables \
  --set-group "Grupo X" \
  --set-members "Integrante 1" "Integrante 2" \
  --set-date 2026-05-28
```

Esperar:
- `deliverables/out/informe.pdf`
- `deliverables/out/pitch.pdf`
- `deliverables/out/pitch.pptx`
- `deliverables/out/entrega_moodle.zip`

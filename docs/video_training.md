# Entrenamiento desde videos etiquetados

Este flujo permite usar videos ya grabados como dataset de entrenamiento, tanto localmente como en Google Colab.

## Layouts soportados

### 1. Por clase

```text
data/train_videos/
  hola/
    clip_01.mp4
    clip_02.mp4
  gracias/
    clip_01.mp4
```

Usar:

```bash
python3 vcpria.py train-videos-stack \
  --videos-root data/train_videos \
  --work-dir runs/demo_videos \
  --layout label
```

### 2. Por sujeto y clase

```text
data/train_videos/
  S1/
    hola/
      clip_01.mp4
    gracias/
      clip_01.mp4
  S2/
    hola/
      clip_01.mp4
```

Usar:

```bash
python3 vcpria.py train-videos-stack \
  --videos-root data/train_videos \
  --work-dir runs/demo_videos \
  --layout subject_label
```

## Qué genera

### `train-videos-stack`

1. Extrae ROIs de mano desde cada video
2. Escribe:
   - `prepared_from_videos/samples.csv`
   - `prepared_from_videos/landmarks.csv`
   - `prepared_from_videos/images_raw/`
   - `prepared_from_videos/images_masked/` si aplica
3. Entrena:
   - baseline de landmarks
   - CNN sobre ROI
4. Genera métricas y reportes en el `work-dir`

### `train-videos-mm-stack`

1. Extrae secuencias multimodales por video:
   - mano izquierda
   - mano derecha
   - pose
   - cara
2. Guarda `.npz` por clip en `prepared_multimodal_from_videos/multimodal_sequences/`
3. Entrena el baseline temporal multimodal

## Recomendaciones prácticas

- Usar clips donde cada video represente una sola seña o palabra objetivo.
- Mantener fondos y ángulos variados si se busca robustez.
- Para evitar leakage, el pipeline usa `group_id = video_id`, así los frames del mismo clip no se mezclan entre train y test.
- Si hay varios participantes, preferir `--layout subject_label`.
- En Mac con Apple Silicon, para CNN conviene usar `--cnn-device mps`.

## Colab

El notebook `/Users/sebastian/Documents/VC-pria/notebooks/vc_pria_colab_video_training.ipynb` está preparado para este flujo.

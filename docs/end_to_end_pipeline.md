# Pipeline end-to-end recomendado

Objetivo: usar `MediaPipe + OpenCV` como nucleo del sistema y entrenar arriba un clasificador liviano o CNN/temporal.

## Arquitectura recomendada

### 1. Captura y preprocesado

- Entrada desde webcam o video.
- OpenCV para:
  - resize y control de FPS,
  - CLAHE / denoise,
  - mascara de piel,
  - tracking temporal del ROI.

### 2. Deteccion de mano

- MediaPipe Hands para obtener:
  - `landmarks` de la mano,
  - `handedness`,
  - `bbox`.

### 3. Dos caminos de clasificacion

#### Pipeline A: landmarks + modelo liviano

- Input: landmarks de MediaPipe.
- Features: distancias, angulos y relaciones geometricas.
- Modelo: SVM.
- Uso ideal:
  - letras,
  - senias estaticas,
  - tiempo real con pocos datos.

#### Pipeline B: ROI + CNN

- Input: recorte de mano generado con OpenCV.
- Modelo: MobileNetV3 small fine-tuned.
- Uso ideal:
  - clases mas visuales,
  - mayor robustez ante variacion de pose/apariencia,
  - comparacion contra baseline geometrico.

#### Pipeline C: secuencias temporales

- Input: ventanas de landmarks en el tiempo.
- Modelo: baseline temporal del repo.
- Uso ideal:
  - senias dinamicas acotadas,
  - pruebas iniciales antes de pasar a modelos temporales mas complejos.

## Flujo local completo

### Opcion 1: dataset propio para demo webcam

Flujo integrado:

```bash
python3 lsupria.py train-stack \
  --csvs data/S1/landmarks.csv data/S2/landmarks.csv \
  --work-dir runs/demo_stack \
  --cnn-image-col img_raw_path \
  --group-col subject_id \
  --cnn-epochs 10
```

Eso genera:
- `runs/demo_stack/data/merged_landmarks.csv`
- `runs/demo_stack/models/landmarks.joblib`
- `runs/demo_stack/models/cnn.pt`
- `runs/demo_stack/results/eval_split.json`
- `runs/demo_stack/results/dataset_stats.md`
- `runs/demo_stack/results/ablation_table.md`
- `runs/demo_stack/results/report.md`

1. Recoleccion

```bash
python3 lsupria.py collect \
  --out data/S1 \
  --subject-id S1 \
  --labels A B C hola gracias si no ayuda
```

2. Entrenamiento landmarks

```bash
python3 lsupria.py train-landmarks \
  --csv data/S1/landmarks.csv \
  --out models/landmarks.joblib
```

3. Entrenamiento CNN

```bash
python3 lsupria.py train-cnn \
  --csv data/S1/landmarks.csv \
  --image-col img_raw_path \
  --out models/cnn.pt
```

4. Evaluacion

```bash
python3 lsupria.py eval-split \
  --csv data/S1/landmarks.csv \
  --landmarks-model models/landmarks.joblib \
  --cnn-model models/cnn.pt \
  --cnn-image-col img_raw_path
```

5. Demo web

```bash
python3 lsupria.py web \
  --landmarks-model models/landmarks.joblib \
  --cnn-model models/cnn.pt \
  --open-browser
```

### Opcion 2: iLSU-T como fuente complementaria

1. Descargar

```bash
python3 lsupria.py ilsut-download \
  --out-dir data/ilsut_archives \
  --sources source2 source3 \
  --skip-existing
```

2. Extraer

```bash
python3 lsupria.py ilsut-extract \
  --archives-dir data/ilsut_archives \
  --out-root data/ilsut_extracted \
  --sources source2 source3 \
  --skip-existing
```

3. Preparar weak labels + entrenar

```bash
python3 lsupria.py ilsut-train \
  --episodes-csv /ABS/PATH/episodes.csv \
  --root data/ilsut_extracted \
  --keywords deliverables/ilsut_keywords.example.json \
  --work-dir data/ilsut_run \
  --pipelines cnn landmarks \
  --preprocess \
  --skin-mask \
  --camera-like
```

## Recomendacion metodologica para el curso

- Para la demo final, priorizar `Pipeline A` y `Pipeline B` sobre dataset propio.
- Usar iLSU-T como extension experimental o comparativa, no como unica fuente de verdad para letras.
- Si van a incluir senias dinamicas, mantener el alcance chico y reportarlo como baseline temporal.

## Resultado esperado

Con este enfoque, el proyecto queda de punta a punta:
- OpenCV procesa y estabiliza la entrada,
- MediaPipe estructura la mano en landmarks + bbox,
- los modelos clasifican,
- la web muestra la prediccion y compone la frase.

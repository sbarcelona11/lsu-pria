# Entrega final (Moodle)

Este directorio contiene plantillas y scripts para generar:

- `deliverables/out/informe.pdf` (informe técnico)
- `deliverables/out/pitch.pdf` (presentación pitch, máximo 8 min)
- `deliverables/out/entrega_moodle.zip` (paquete para subir)

Generación (desde la raíz del repo):

```bash
source .venv/bin/activate

# Dependencias para generar PDFs/PPTX (incluye fpdf2)
python3 -m pip install -r requirements.txt

# Build completo + checks + zip
python3 lsupria.py deliverables
```

Si querés que el informe/pitch incluyan una sección automática con métricas SLT, podés pasar un `summary.json`:

```bash
python3 lsupria.py deliverables --slt-summary-json runs/whisperx_slt_gen/summary.json
```

Notas:
- Si tu `pip` está deshabilitado por el Python del sistema, usá el intérprete del venv explícitamente:
  - `./.venv/bin/python -m pip install -r requirements.txt`
  - `./.venv/bin/python lsupria.py deliverables ...`

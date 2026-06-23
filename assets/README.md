# assets/

Place the following figures here, extracted from the paper PDF:

| Filename | Paper reference | Content |
|---|---|---|
| `pipeline.png` | Fig. 1 | Three-stage training and compression pipeline diagram |
| `qualitative.png` | Fig. 2 | Qualitative retrieval results (two BLT sandwich images) |

## How to extract

Open the PDF in any PDF viewer and use a screenshot tool, or use a PDF-to-image
converter such as:

```bash
pip install pdf2image
python - <<'EOF'
from pdf2image import convert_from_path
pages = convert_from_path("TinyVLM_Full_Paper.pdf", dpi=200)
pages[0].save("pipeline_page.png")   # figure is on page 1
pages[1].save("results_page.png")    # figure 2 is on page 2
EOF
```

Crop each figure from the saved page images and save with the filenames above.

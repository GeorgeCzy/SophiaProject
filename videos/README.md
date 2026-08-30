This folder contains the study videos used by the rating website.

Use one subfolder per experiment version:

- `complete/`: the full-system version.
- `single-llm/`: the single-LLM ablation version.

Each experiment version has its own manifest in `../manifests/`. If video
filenames change, update that experiment's manifest so each item points to the
correct clip.

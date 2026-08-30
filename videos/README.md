This folder contains the study videos used by the rating website.

Use one subfolder per experiment version:

- `complete/`: the current full-system version.
- `single-llm/`: the single-LLM version once its videos are added.

Each experiment version has its own manifest in `../manifests/`. If video
filenames change, update that experiment's manifest so each item points to the
correct clip.

# Repository Guidelines

## Project Structure & Module Organization

This is a Python/PyTorch multimodal model repository. Core architectures live in `model/`; dataset loading and preprocessing are in `dataset/`; distributed fine-tuning entry points and helpers are in `trainer/`. Use `eval_omni.py` for command-line inference, `scripts/` for model conversion and the Gradio demo, and `webui/` for the Flask-based interface. Evaluation media is under `dataset/eval_omni/`, while documentation images belong in `images/`. Keep downloaded encoders under `model/` and generated weights under `out/` or `checkpoints/`; do not commit large model artifacts.

## Setup, Run, and Training Commands

- `pip install -r requirements.txt` installs project dependencies. Install the appropriate PyTorch, torchvision, and torchaudio builds separately for your CUDA or CPU environment.
- `python eval_omni.py --load_from model --weight sft_omni` runs local command-line inference with project-format weights.
- `cd scripts; python web_demo_omni.py` launches the Gradio demo after a Transformers-format model is placed under `scripts/`.
- `cd trainer; bash train.sh` runs the documented single-GPU mini-data training pipeline. Review paths, GPU selection, and WandB settings before starting.
- `python -m compileall model dataset trainer scripts webui eval_omni.py` provides a quick syntax check. There is no separate build step.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for functions, variables, modules, and CLI flags; use `PascalCase` for classes and configuration types. Group standard-library, third-party, and local imports. Prefer small, explicit functions and preserve existing PyTorch tensor-shape conventions. No formatter or linter is configured, so avoid unrelated reformatting.

## Testing Guidelines

The repository currently has no automated test suite or coverage gate. For each change, run the syntax check and the smallest relevant inference, conversion, dataset, or training smoke test. New pure-logic tests should go in `tests/` and follow `test_<module>.py`; document any required GPU, weights, or sample data in the pull request.

## Commit & Pull Request Guidelines

Recent history uses short Chinese-language commit subjects, often for note or branch updates. Keep each commit focused and use a specific imperative subject, for example `Fix audio projection training`. Pull requests should explain the purpose and behavioral impact, list verification commands, link related issues, and include screenshots for WebUI changes or concise logs/metrics for training changes. Never include downloaded weights, credentials, or private dataset paths.

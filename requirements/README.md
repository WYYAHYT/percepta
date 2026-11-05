# Per-model dependency sets, installed into isolated virtualenvs.
#
# Each file here describes ONE model's environment. This is how two models that need
# incompatible torch/numpy versions coexist without containers — see
# docs/model-isolation.md.
#
# Usage:
#   uv venv .venvs/<model> -p 3.10
#   uv pip install --python .venvs/<model>/bin/python -r requirements/<model>.txt
#
# Rules for every file in this directory:
#   * pin exact versions (no bare `>=`) — these environments are deployment artifacts;
#   * include the model's own framework (torch etc.);
#   * always pin numpy within the range the model supports;
#   * remember the worker also needs `numpy` and `pydantic` (percepta's schema/protocol).

set -euo pipefail
sed '/^\[tool\.poetry\.dependencies\]$/a\torch = {source = "pytorch-cpu"}' \
  pyproject.toml \
  > github-poetry/pyproject.toml
cd github-poetry
poetry lock

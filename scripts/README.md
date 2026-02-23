# Scripts

## `normalize_notebooks.py`

Fixes Jupyter notebook outputs so they pass nbformat validation (e.g. on GitHub).

**Run manually:**

```bash
# All notebooks in notebooks/
python scripts/normalize_notebooks.py

# Specific file(s)
python scripts/normalize_notebooks.py notebooks/01_eda.ipynb
```

**Run automatically before each commit:**

1. Install pre-commit: `pip install pre-commit`
2. Install the git hook: `pre-commit install`
3. On each commit, staged `.ipynb` files are normalized automatically.

To run the hook on all notebooks without committing: `pre-commit run normalize-notebooks --all-files`.

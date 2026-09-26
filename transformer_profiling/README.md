# Transformer profiling tools

## Python environment

The repository's `environment.yml` defines `gem5_env` with Python 3.9.23.
Its pip dependencies and the root `requirements.txt` include pandas 2.2.3
and Plotly 6.3.0 for interactive reports.

For an **existing** environment, install only the plotting dependencies from
the repository root; there is no need to recreate the environment or reinstall
the full PyTorch/CUDA dependency set:

```bash
# eslsrv12; use your own Conda installation path on another machine.
source "$HOME/i2ce/miniforge3/etc/profile.d/conda.sh"
conda activate gem5_env
python -m pip install -r transformer_profiling/requirements-visualization.txt
python -c "import pandas, plotly; print('pandas:', pandas.__version__); print('Plotly:', plotly.__version__)"
python -m pip check
```

`pip check` validates dependencies of installed distributions. The import
check above separately confirms that both plotting packages are available to
the active Python interpreter. Jupyter is optional.

`add_experiment.py` uses only the standard library; collecting a gem5 run
does not require pandas or Plotly. See [final/README.md](final/README.md) and
[the experiment runner](../tools/exp/README.md) for the collection workflow.

# Synthesis Planning with PRMs

## Installation

`conda create -n synthesis python==3.10`

```shell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python -m pip install -r requirements.txt
```

`sh scripts/download.sh`
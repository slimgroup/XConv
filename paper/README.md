# Reproducible results for the NeurIPS21 submission

THis directory contains the scripts to reproduce the results presented in the 20201 BeurIPS conference submission. Each script name contains the Figure or Table number to know which script corresponds to which results. 

## Setup

Each script requires to have the packge `pyxconv` installed via

```bash
pip install -e .
```

in the top directory, or added in the julia path for the julia package.

```julia
]add path/to/top/directory
```

## Run

Each script is standalone, and is producing the figure needed, via `julia scriptname` or `python scriptname`.

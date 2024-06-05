# Using FOB with SMAC for HPO
Run all commands from the root of the FOB repository.

## Setup
```bash
conda create -n fob-hpo python=3.10 -y
conda activate fob-hpo
pip install -r requirements.txt
pip install -e .
pip install -r examples/smac/requirements.txt
```

## Example
```bash
python examples/smac/optimizer_comparison.py
```

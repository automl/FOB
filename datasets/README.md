## TODO
- clean up after installation (tar.gz) for cifar 10
- clean up after installation (tar.gz) for cifar 100
- clean up after installation (downloads?) for mnist?

## Installation

create a conda env and use it

```
conda create --name bob python=3.10
conda activate bob
```

packages installed might be in non optimal order:

I installed it in this order, but absl-py installs 2.0 and tensoorflow will later downgrade absl to 1.4.0
```
pip install absl-py
pip install tensorflow-datasets
```
the 3rd package needed already differs for the cuda version we want to use:  
https://pytorch.org/get-started/locally/ 
i used the following
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```
pip install tensorflow
```

## Usage

cd into folder with file, give root where to install, choose workloads, --all to get all of them
```
python dataset_setup.py --data_dir ~/data_bob --all
```

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
Later we will have a requirements.txt;
here is how to manually install the packages, depending on your compute platform

PyTorch 2.1.1 CUDA 12.1
```
pip3 install torch torchvision torchaudio
pip install lightning
```

PyTorch 2.1.1 CUDA 11.8
```
# untested
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning
```


## Usage

cd into folder with file, give root where to install, choose workloads, --all to get all of them
```
python dataset_setup.py --data_dir ~/data_bob --all
```

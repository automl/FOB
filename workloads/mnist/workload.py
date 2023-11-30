# TODO: best way to import? just put in same file maybe...
from DataModule import MNISTDataModule

def get_datamodule():
    my_path = "TODO/data/mnist"
    mnist = MNISTDataModule(my_path)

def get_model():
    pass

# auxilliary information like batch_size, max_steps, etc...
def get_specs():
    pass

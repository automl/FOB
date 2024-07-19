from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    from pytorch_fob.tasks.mnist.data import MNISTDataModule
    from pytorch_fob.tasks.mnist.model import MNISTModel


class Dataset(BaseDataset):
    name = 'MNIST'

    parameters = {
        'num_hidden': [10],
        'activation': ['Sigmoid', 'ReLU'],
        'seed': [42, 47]
    }

    def get_data(self):
        model = MNISTModel(
            num_hidden=self.num_hidden,
            activation=self.activation
        )

        data_dir = get_data_path('mnist')
        data_module = MNISTDataModule(
            data_dir=data_dir, seed=self.seed
        )

        return dict(model=model, data_module=data_module)

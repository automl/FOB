# Tasks
We provide a set of tasks to train and evaluate models. A task consists of a model and a dataset.

We currently have the following tasks:

TODO: table from main when its final.

Each task has their own `README.md` file with more details.

## Adding your own task
To add your own task, you need to create a subfolder in the `tasks` directory. The name of that folder will be the name used to invoke the task. Within the folder you need to provide the following files: `task.py`, `model.py`, `data.py`, `default.yaml` and `README.md`. 

There is a [template](template) task with useful comments, which can be used as a starting point.

### data.py
Here you provide the code for interacting with your dataset. As we use [lightning](https://lightning.ai/docs/pytorch/stable/), you will need to crate a [LightningDataModule Datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).  

The class you create must inherit from `TaskDataModule` which in turn inherits from `LightningDataModule`. The base `TaskDataModule` already defines some default methods for the dataloader methods, so if you do not need any custom dataloaders you can probably leave them.

The two methods you need to implement are [prepare_data](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data) and [setup](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup). In `prepare_data` you need to put you downloading and data preprocessing logic. In `setup` you should load and split your dataset and set the `self.data_train, self.data_val, self.data_test` attributes in the appropriate stages.

### model.py
Here you provide the code for the model. As we use [lightning](https://lightning.ai/docs/pytorch/stable/), you will need to create a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).  

The class you create must inherit from `TaskModel` which in turn inherits from `LightningModule`. The `__init__` method should have the following signature:  
```python
def __init__(self, optimizer: Optimizer, config: TaskConfig):
```
In the `__init__` method you need to create your model, and pass it to the `super().__init__` call. There the model is wrapped into a `GroupedModel` which splits the model parameters into weight_decay and non-weight_decay groups. If you want to specify your own parameter groups (e.g. for different learning rates) you need to wrap your model in a `GroupedModel` yourself, before passing it to the `super().__init__` call.

The other methods you neet to implement are [training_step](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step), [validation_step](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step) and [test_step](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step). Here you need to implement the training and evaluation logic.

### task.py
Here you only need to provide two simple functions:
```python
def get_datamodule(config: TaskConfig) -> TaskDataModule
```
which returns an instance of your `DataModule` class, and
```python
def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]
```
which returns an instance of your `TaskModel` class and an instance of your `DataModule` class.

### default.yaml
Here you can provide default values for all the hyperparameters your task needs. All keys under the `task` section will be added to the `TaskConfig`. 

There are some required parameters you need to specify:
```yaml
task:
  name: my_awesome_task    # same as directory name
  batch_size: 123
  max_epochs: 42
  max_steps: null          # should be left null, use max_epochs instead
  target_metric: val_acc   # choose a metric that is being logged in your LightningModule
  target_metric_mode: min  # min or max 
engine:
  devices: 1               # number of devices to use
  sbatch_args:
    time: 00:05:00         # estimated time to train
evaluation:
  plot:
    metric: test_acc
    test_metric_mode: min
    format: "2.1"
    limits: [0, 100]       # colorbar limits
optimizer:
  name: adamw_baseline     # set the default optimizer
```

You can optionally set and override optimizer defaults, e.g.:
```yaml
optimizer:
  name: adamw_baseline
  learning_rate: 0.1
```
would use a default learning rate of 0.1 instead of the one specified in the `default.yaml` of the optimizer. Note that this applies to all optimizers. So if the user chooses a different optimizer, they will still get the default learning rate specified here.

### README.md
Here you should provide a short description of your task, and a baseline performance. Follow the template as seen in the existing tasks.

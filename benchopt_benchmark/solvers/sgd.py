from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from pytorch_fob.optimizers.sgd_baseline.optimizer import \
        configure_optimizers
    from pytorch_fob.optimizers.optimizers import OptimizerConfig
    from pytorch_fob.engine.parameter_groups import GroupedModel
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from lightning import Trainer, Callback


class Optimizer():
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def configure_optimizers(self, model: GroupedModel) -> OptimizerLRScheduler:
        return configure_optimizers(model, self.config)


class Solver(BaseSolver):
    name = 'SGD'

    parameters = {
        'learning_rate': [1e-3],
        'weight_decay': [1e-4],
        'momentum': [0.9],
        'nesterov': [True],
        'max_steps': [200],
        'eta_min_factor': [0.1],
        'lr_interval': ['step'],
        'batch_size': [64]
    }
    sampling_strategy = 'run_once'

    def set_objective(self, model, data_module):
        self.model = model
        self.data_module = data_module

        self.data_module.set_batch_size(self.batch_size)

        config = OptimizerConfig(
            optimizer_key='sgd',
            task_key='benchopt',
            config=dict(
                sgd=dict(
                    name=self.name,
                    lr_interval=self.lr_interval,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                    eta_min_factor=self.eta_min_factor,
                ),
                benchopt=dict(
                    max_steps=self.max_steps,
                    max_epochs=30,
                ),
            )
        )

        optimizer = Optimizer(config)
        self.model.set_optimizer(optimizer)

    def run(self, _):
        # class BenchoptCallback(Callback):
        #     def on_train_epoch_end(self, trainer, pl_module):
        #         trainer.should_stop = not cb()

        self.trainer = Trainer(
            max_epochs=3,
            # callbacks=[BenchoptCallback()]
        )
        self.trainer.fit(self.model, self.data_module)

    def get_result(self):
        return dict(trainer=self.trainer)

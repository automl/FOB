from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    from lightning import Trainer


class Objective(BaseObjective):
    name = "FOB"

    requirements = [
        "pip::git+https://github.com/automl/FOB.git"
    ]

    def set_data(self, model, data_module):
        self.model = model
        self.data_module = data_module

    def evaluate_result(self, trainer: Trainer):
        score_train = trainer.validate(self.model, datamodule=self.data_module)
        score_val = trainer.validate(self.model, datamodule=self.data_module)
        score_test = trainer.test(self.model, datamodule=self.data_module)
        return dict(
            **{f'train_{k}': v for k, v in score_train[0].items()},
            **{f'val_{k}': v for k, v in score_val[0].items()},
            **{f'test_{k}': v for k, v in score_test[0].items()},
            value=score_val[0]['val_loss'],
        )

    def get_objective(self):
        return dict(
            model=self.model,
            data_module=self.data_module
        )

    def get_one_result(self):
        return dict(trainer=Trainer(
            devices=self.devices,
            enable_progress_bar=True,
        ))

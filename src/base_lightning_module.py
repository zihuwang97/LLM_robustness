import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import warnings


def _get_loss_value(loss):
    if isinstance(loss, dict):
        return loss['loss']
    else:
        return loss


class base_lightning_module(LightningModule):
    def training_step(self, batch, batch_idx):
        if hasattr(self, "training_and_validation_step"):
            loss = self.training_and_validation_step(**batch)
            self.log("train_loss", _get_loss_value(loss), on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            raise NotImplementedError("Define training_and_validation step in subclass")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if hasattr(self, "training_and_validation_step"):
            loss = self.training_and_validation_step(**batch)
            self.log("val_loss", _get_loss_value(loss), on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            raise NotImplementedError("Define training_and_validation step in subclass")

    def _checkpoint(self):
        return ModelCheckpoint(monitor="val_loss", save_weights_only=True)

    def _load_best_checkpoint(self, msg="loading"):
        best_model_path = self._checkpoint.best_model_path
        best_model_score = self._checkpoint.best_model_score
        if best_model_score is not None:
            print(f"{msg} checkpoint {best_model_path} with score {best_model_score}")
            self.load_state_dict(torch.load(best_model_path)['state_dict'])
        else:
            warnings.warn("No checkpoints found!")

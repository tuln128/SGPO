import os
import time
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from models.pretraining.callbacks import HuggingFaceCheckpointer
import hydra

### Modified from https://github.com/ngruver/NOS/blob/main/seq_models/trainer.py ###
class BaseModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.discr_batch_ratio = None
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    def training_step(self, batch):        
        out = self.forward(batch
            #labels=batch["labels"] if "labels" in batch else None,
        )        

        log_dict = {f"train_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict)  # Don't seem to need rank zero or sync dist

        # if "labels" in batch:
        #     if self.discr_batch_ratio is None:
        #         out["loss"] = out["loss"] + out["regression_mse"]
        #     elif batch_idx % self.discr_batch_ratio == 0:
        #         out["loss"] = out["regression_mse"]
        
        return out["loss"]

    def validation_step(self, batch):
        with torch.no_grad():
            out = self.forward(batch
                #labels=batch["labels"] if "labels" in batch else None,
            )
                
        log_dict = {f"val_{k}" : v for k, v in out.items()}
        self.log_dict(log_dict, rank_zero_only=True)

        return {"val_loss": out['loss']}

    def configure_optimizers(self):
        config = {
            "optimizer": self.opt
        }

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() #avoid lr=0 at start for warmup

            config["lr_scheduler"] = {
                "scheduler": self.lr_scheduler,
                "frequency": 1,
                "interval": "epoch",    # Call after 1 epoch
            }

        return config
    
def get_trainer(config):
    save_path = os.path.join("checkpoints", config.pretrain_model.name, config.data.name)

    if "causalLM" in config.pretrain_model.name:
        hf_callback = HuggingFaceCheckpointer(
            save_dir=os.path.join(save_path), #"huggingface"
            #tokenizer=None, 
            save_every_n_epochs=1  # Save every epoch
        )
        callbacks = [hf_callback]
    elif "mdlm" in config.pretrain_model.name or "udlm" in config.pretrain_model.name:
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename="best_model", monitor="val/ppl", mode="min", save_top_k=1, save_last=True) #keep the model with the lowest validation loss and the last one
        callbacks = [checkpoint_callback]
    else:
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, filename="best_model", monitor="val_loss", mode="min", save_top_k=1, save_last=True) #only keep the model with the lowest validation loss
        callbacks = [checkpoint_callback]
    
    experiment_name = config.exp_name + "-" + config.pretrain_model.name + "-" + config.data.name
    wandb_logger = WandbLogger(experiment_name, project=config.wandb_project) #dir=config.exp_name

    accelerator, strategy = "cpu", None
    if config.pretrain_model.train.ngpu > 0:
        accelerator = "gpu"
        strategy = "ddp"

    # Override strategy if specified in the model config
    if hasattr(config.pretrain_model, 'strategy'):
        strategy = config.pretrain_model.strategy

    if "mdlm" in config.pretrain_model.name or "udlm" in config.pretrain_model.name:
        trainer = hydra.utils.instantiate(config.pretrain_model.trainer, logger=wandb_logger, callbacks=callbacks, accelerator=accelerator, strategy=strategy)
    else:
        trainer = pl.Trainer(
            default_root_dir=config.exp_name,
            gradient_clip_val=config.pretrain_model.train.gradient_clip,
            min_epochs=config.pretrain_model.train.min_epochs,
            max_epochs=config.pretrain_model.train.max_epochs,
            val_check_interval=config.pretrain_model.train.val_interval, #can use fractions
            callbacks=callbacks,
            logger=wandb_logger,
            log_every_n_steps=config.pretrain_model.train.log_interval,
            accelerator=accelerator,
            strategy=strategy,
            devices=config.pretrain_model.train.ngpu,
            enable_progress_bar=True,
        )

    return trainer
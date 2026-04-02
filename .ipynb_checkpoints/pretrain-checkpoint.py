import os
import sys
import pprint
from pathlib import Path
import hydra
import torch
import wandb
import warnings
from omegaconf import OmegaConf
import sys
import random
import numpy as np

from models.pretraining.trainer import get_trainer
from models.pretraining.dataset import (
    get_loaders,
)
from models.pretraining.model.d3pm_evodiff import ByteNetDiffusion
from models.pretraining.model.continuous_diffusion import GaussianDiffusion
from util.seed import set_seed

@hydra.main(config_path="configs", config_name="pretrain_config")
def main(config):
    Path(config.exp_name).mkdir(parents=True, exist_ok=True)
    
    root_dir = hydra.utils.get_original_cwd()
    os.chdir(root_dir)     # Change back to the root directory

    model = hydra.utils.instantiate(config.pretrain_model.model, _recursive_=False) #_recursive_=False

    #TODO: could continue training from checkpoint
    #Probably not necessary if model training is fast

    # if config.ckpt_path is not None:
    #     state_dict = torch.load(config.ckpt_path)['state_dict']
    #     model.load_state_dict(state_dict)
    if config.pretrain_model.train.ngpu > 0:
        model.to(torch.device("cuda"))

    train_loader, validation_loader = get_loaders(config)

    #TODO: currently not fully determinsitic, but the weights are initialized with a seed
    set_seed(config.pretrain_model.train.seed)

    if True: #training diffusion with random timesteps
        #model.freeze_for_discriminative()
        trainer = get_trainer(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # catch really annoying BioPython warnings
            
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader,
            )

            #save the lightning checkpoint as a state dict of the network that is compatible with EvoDiff - let's just do this during loading instead
            # save_path = os.path.join("checkpoints", config.pretrain_model.name, config.data.name)
            # best_checkpoint = os.path.join(save_path, "best_model.ckpt")
            
            # if "d3pm" in config.pretrain_model.name:
            #     lightning_model = ByteNetDiffusion.load_from_checkpoint(best_checkpoint)
            # elif "continuous" in config.pretrain_model.name:
            #     lightning_model = GaussianDiffusion.load_from_checkpoint(best_checkpoint)
            # torch.save(lightning_model.network.state_dict(), os.path.join(save_path, "best_model_state_dict.pth"))

    # else:
    #     train_dl, valid_dl = [
    #         make_discriminative_loader(config, model, dl) for dl in [train_dl, valid_dl]
    #     ]

if __name__ == "__main__":    
    main()
    sys.exit()
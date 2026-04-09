import os
from omegaconf import OmegaConf
import pickle
import pandas as pd
import hydra
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
import wandb
from models.pretraining.collaters import collate_fn_mapping
from util.seed import set_seed  
import tqdm
# from dataset.protein import ProteinPredictorDataset
from sampling.DPO import DPOInpaint
from sampling.cls_guidance import Classifier_Guidance_Inpaint, Classifier_Guidance_Continuous

def sample(config, n_samples, algorithm, dataset, batch_size, round, unique_only=False, BO=False):
    #calculate number of batches
    max_iterations = n_samples // batch_size
    sample_steps = range(0, max_iterations)
    
    #TODO ideally should be updated into a while loop, but prior is diverse enough that this is not an issue
    # for i in batch_steps:
    for step in tqdm.tqdm(sample_steps):
        _, detokenized = algorithm.inference(num_samples=batch_size, detokenize=True)
        _ = dataset.update_data(detokenized, n_samples, round=round, unique_only=unique_only, BO=BO)
    
    return dataset

@hydra.main(version_base="1.3", config_path="configs", config_name="sample_config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'causalLM' in config.model.name:
        problem_config = OmegaConf.load('configs/problem/protein_DPO.yaml')
    else:
        if "continuous" in config.model.name:
            problem_config = OmegaConf.load('configs/problem/protein_classifier_continuous.yaml')
        else:
            problem_config = OmegaConf.load('configs/problem/protein_classifier_discrete.yaml')

    exp_dir = os.path.join(problem_config.exp_dir, config.data.name, config.pretrained_ckpt.split('/')[0], "prior_sample")
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    data_config = config.data

    # ── CHANGE 1: resolve seq_len from full_seq if available ─────
    if data_config.full_seq is not None:
        if isinstance(data_config.full_seq, (list, tuple)):
            seq_len = len(data_config.full_seq[0])
        else:
            seq_len = len(data_config.full_seq)
        OmegaConf.update(config, "data.seq_len", seq_len)
        print(f"seq_len overridden from full_seq: {seq_len}")
    else:
        seq_len = data_config.seq_len
        print(f"seq_len from config: {seq_len}")
    # ─────────────────────────────────────────────────────────────

    sample_batch_size = config.num_samples if problem_config.sample_batch_size > config.num_samples else problem_config.sample_batch_size
    recursive = False if config.model.name == "mdlm" or config.model.name == "udlm" else True

    ### Sample the initial random data to be used for everything ###
    for task, n_max_mutations in zip(["unconstrained"], [None]):
        set_seed(config.seed)
        save_dir = os.path.join(exp_dir, task)
        os.makedirs(save_dir, exist_ok=True)
    
        fasta_save_path = os.path.join(save_dir, f'generated.fasta')

        #run sampling for the initial round (with repeats)
        data_config = config.data
            
        if 'causalLM' in config.model.name:
            net = instantiate(config.model.model, load_ref_model=False, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device) 
            #initiate algo once, doesn't need to be changed for DPO
            algorithm = DPOInpaint(net=net, data_config=data_config, n_max_mutations=n_max_mutations)
        else:   
            net = instantiate(config.model.model, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device, _recursive_=recursive) 
            if "continuous" in config.model.name:
                algorithm = Classifier_Guidance_Continuous(net=net, data_config=data_config, temperature=0.,n_max_mutations=n_max_mutations)
            else:
                algorithm = Classifier_Guidance_Inpaint(temperature=0., n_max_mutations=n_max_mutations, net=net, data_config=data_config) #dummy to get the project_fn, replace later
       
        dataset = instantiate(problem_config.data, data_config=data_config, tokenizer=net.tokenizer)
        sample(config, config.num_samples, algorithm, dataset, sample_batch_size, 0)
        
        #save dataset.seqs to fasta
        with open(fasta_save_path, 'w') as f:
            for j, seq in enumerate(dataset.seqs):
                f.write(f">{j}\n")
                f.write(f"{seq}\n")
        print(f"Saved generated sequences to {fasta_save_path}")

        #loop through for repeated sampling as a basline for the iterative experiment
        summary_df = pd.DataFrame()
        save_path = os.path.join(exp_dir, 'summary.csv')
        sample_batch_size = config.num_samples_per_round if problem_config.sample_batch_size > config.num_samples_per_round else problem_config.sample_batch_size

        for i in range(config.n_repeats):
            
            for round in range(config.n_rounds + 1):
                set_seed(config.seed + i + round)
                
                print(f'Sampling prior values')
                data_config = config.data

                dataset = instantiate(problem_config.data, data_config=data_config, tokenizer=net.tokenizer)
                sample(config, config.num_samples_per_round, algorithm, dataset, sample_batch_size, round, unique_only=True, BO=True) #make sure everything is unique

                dataset.summary_df["task"] = task
                dataset.summary_df["repeat"] = i
                dataset.summary_df["round"] = round
                dataset.summary_df["method"] = "prior_baseline"
                summary_df = pd.concat([summary_df, dataset.summary_df], axis=0)
                summary_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
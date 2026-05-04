#hpc #tutorial

**Table of Contents**
```table-of-contents
```

# Syntax
Everything in Torch requires specification of the `--account` ID. This can be found in: [projects.hpc.nyu.edu](https://projects.hpc.nyu.edu) -> allocations ->`slurm_account_name`. This tutorial just has mine.
### Submit SBATCH Job
```bash
sbatch <FILE_PATH.EXT>
```
See full documentation [here](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_main_commands/#sbatch).

Example sbatch script
```sbatch
#!/bin/bash
#SBATCH --job-name=SCSI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --output=/scratch/cm6627/.../%j.out
#SBATCH --error=/scratch/cm6627/.../%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cm6627@nyu.edu

export HF_HOME=/scratch/<NetID>/.cache/huggingface

singularity exec --nv \
  --overlay /scratch/cm6627/.../overlay-25GB-conda.ext3:rw \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash -c "
    source /ext3/miniconda3/bin/activate
    cd /scratch/cm6627/fine-tune
    python train_gemma3.py
"
```


### Interactive Job
```bash
srun --gres=gpu:1 --cpus-per-task=8 --pty bash
```
See full documentation [here](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_main_commands/#srun). Note that `--pty bash` needs to go at the end.

| Flag                | Meaning             |
| ------------------- | ------------------- |
| `--gres=gpu:1`      | Request any `1` GPU |
| `--cpus-per-task=8` | Request 8 cpus      |

# First Time User
These are more tricks that I found useful

1. **`--account`**: Everything in Torch requires specification of the `--account` ID. Which can be found in: [projects.hpc.nyu.edu](https://projects.hpc.nyu.edu) -> allocations ->`slurm_account_name`. 
   
You can setup a default account, go to `~/.bashrc` and put
```bash
export SLURM_ACCOUNT="torch_pr_1031_general" 
export SBATCH_ACCOUNT=${SLURM_ACCOUNT}
export SALLOC_ACCOUNT=${SLURM_ACCOUNT}
```
Then run 
```
source ~/.bashrc
```

This file assumes you've done this!

2. **Package Manager:** I don't like using singularity + conda, it's too much of a headache. Instead you can use `uv`. To install find the `wget` command [here](https://docs.astral.sh/uv/getting-started/installation/). 
   You then need to change the UV cache. In `~/.bashrc` add `export UV_CACHE_DIR=/scratch/cm6627/.uv-cache`

   
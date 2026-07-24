# Running `fss` on a GPU at OSC (Cardinal / H100)

Helpers to run the few-shot segmentation pipeline on an OSC **Cardinal** H100 GPU
node (80 GB VRAM — comfortably fits the `large` DINOv3 + SAM 3 in `bfloat16`).

The pipeline auto-detects CUDA, so "running on GPU" is just a matter of (a) a
conda env with a CUDA build of PyTorch + `transformers>=5.12`, and (b) a job that
requests a GPU. These scripts do both.

| Script | Where to run | What it does |
|--------|--------------|--------------|
| [setup_env.sh](setup_env.sh) | login node | Create the conda env, install CUDA PyTorch + `fss` |
| [download_weights.sh](download_weights.sh) | login node | Pre-cache DINOv3 + SAM 3 weights into scratch HF cache |
| [run_gpu.slurm](run_gpu.slurm) | `sbatch` | Run `python -m fss` on 1 H100 |

## Cluster facts (auto-detected)

- Account/project: **PAS2699**; GPU partitions: `gpu` (7-day), `debug` (1-hour, for tests).
- GPUs: `h100` (request `--gpus-per-node=1`).
- HF cache lives on scratch: `/fs/scratch/PAS2699/$USER/hf_cache` (large, regenerable).
- Conda comes from `module load miniconda3/24.1.2-py310`.

## One-time setup (login node)

```bash
cd ~/job-repository/few-shot-segmentation

bash osc/setup_env.sh            # ~5 min: builds the `fss` conda env

# The facebook/dinov3-* and facebook/sam3 weights are GATED — log in once and
# click "Agree and access" on each model's Hugging Face page first:
huggingface-cli login
bash osc/download_weights.sh     # pre-caches weights to scratch (login = internet)
# larger backbone instead of the default base:
# bash osc/download_weights.sh facebook/dinov3-vitl16-pretrain-lvd1689m
```

## Run as a batch job

Edit the input paths in [run_gpu.slurm](run_gpu.slurm) (or pass them at submit
time), then:

```bash
sbatch --export=ALL,QUERY=q.png,OUT=out.png,\
SUPPORT="s1.png:m1.png s2.png:m2.png",DINOV3=large \
    osc/run_gpu.slurm

squeue --me                      # watch the queue
tail -f fss_<jobid>.log          # follow progress; outputs: out.png + debug.png
```

## Run interactively (handy for debugging)

```bash
# request one H100 for an hour
sinteractive -A PAS2699 -p gpu -g 1 -t 01:00:00
# (equivalently: salloc -A PAS2699 -p gpu --gpus-per-node=1 -t 01:00:00)

module load miniconda3/24.1.2-py310
source "$MINICONDA3_HOME/etc/profile.d/conda.sh"
conda activate fss
export HF_HOME=/fs/scratch/PAS2699/$USER/hf_cache

python -m fss --support s1.png:m1.png --query q.png --out out.png \
    --dinov3 base --dtype bfloat16 --debug-out debug.png
```

## Notes

- **`--dtype bfloat16`** runs the forward passes under `torch.autocast` (weights
  stay fp32) — fast and numerically stable on H100. Use `float32` if you want bit
  exact CPU/GPU parity; `auto` picks bf16 on CUDA automatically.
- The batch script sets `HF_HUB_OFFLINE=1` so the GPU node never reaches the
  network — it relies on the cache populated by `download_weights.sh`. If the
  Cardinal compute nodes have outbound internet for you, you can unset it.
- VRAM: even `large` + SAM 3 in bf16 uses well under 20 GB, so a single H100 is
  ample; you can also request `debug` for short runs.

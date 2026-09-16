"""
ALCF Compute Finder
===================
A guide for Argonne Leadership Computing Facility users: pick the right system
for your workload, then get the exact commands to run your first job.

Run with:  streamlit run app.py

-----------------------------------------------------------------------------
SPEC PROVENANCE
-----------------------------------------------------------------------------
Every number below was checked against ALCF's own pages on the date in
LAST_VERIFIED, which the UI shows so users know how stale the data may be.
Sources are recorded per system in SYSTEMS[...]["sources"]. Primary references:

  Aurora     https://www.alcf.anl.gov/aurora   + https://docs.alcf.anl.gov/aurora/
  Polaris    https://www.alcf.anl.gov/polaris  + https://docs.alcf.anl.gov/polaris/
  Sophia     https://docs.alcf.anl.gov/sophia/
  Crux       https://docs.alcf.anl.gov/crux/   + https://www.alcf.anl.gov/crux
  Tara       https://www.alcf.anl.gov/tara
  Minerva    https://www.alcf.anl.gov/minerva
  Janus      https://www.alcf.anl.gov/janus
  AI Testbed https://docs.alcf.anl.gov/ai-testbed/ + https://www.alcf.anl.gov/alcf-ai-testbed
  Inference  https://docs.alcf.anl.gov/services/inference-endpoints/

Hardware and queue policies change. Re-check before making capacity decisions.
"""

import pandas as pd
import streamlit as st

DOCS = "https://docs.alcf.anl.gov"
WWW = "https://www.alcf.anl.gov"
LAST_VERIFIED = "August 21, 2026"

st.set_page_config(
    page_title="ALCF Compute Finder",
    page_icon="◧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Workload vocabulary
#
# Systems score 0-3 against each task. 3 = this is what the machine is for,
# 0 = wrong tool. These weights drive both the ranking and the "why" text.
# ---------------------------------------------------------------------------
TASKS = {
    "Use an LLM through an API (no setup)": "llm_api",
    "Train or fine-tune a model": "train",
    "Run inference at scale": "inference",
    "Large-scale simulation (MPI / CFD / MD / QCD)": "simulation",
    "Data analysis, preprocessing, workflows": "data",
    "Learn HPC, teach a class, run notebooks": "learning",
    "Benchmark or port code to novel AI hardware": "novel_hw",
    "Visualization": "viz",
}

SCALES = {
    "Part of one accelerator": 0,
    "One node": 1,
    "2-10 nodes": 10,
    "10-100 nodes": 100,
    "Hundreds to thousands of nodes": 10000,
}

ACCEL_CHOICES = [
    "No preference",
    "NVIDIA (CUDA)",
    "Intel (SYCL / oneAPI)",
    "Non-GPU AI accelerator",
    "CPU only",
]

# ---------------------------------------------------------------------------
# SYSTEMS
#
# cores_per_node   = PHYSICAL cores (threads tracked separately, since the
#                    original app used thread counts as if they were cores)
# cpu_mem_gb       = per-node CPU memory
# accel_mem_gb     = per-node accelerator memory
# node_storage     = node-local scratch (free text, since units differ)
# facility_storage = shared parallel file systems (NOT the same thing; the
#                    original app mixed these two in one "storage_tb" field)
# ---------------------------------------------------------------------------
SYSTEMS = [
    {
        "key": "aurora",
        "name": "Aurora",
        "family": "Supercomputer",
        "status": "production",
        "tagline": "Exascale Intel GPU system. The one you use when nothing smaller fits.",
        "vendor": "Intel / HPE Cray EX",
        "nodes": 10624,
        "cpu": "2x Intel Xeon CPU Max (Sapphire Rapids w/ HBM)",
        "cores_per_node": 104,
        "threads_per_node": 208,
        "cpu_mem_gb": 1024,
        "cpu_mem_note": "1,024 GB DDR5 + 128 GB CPU HBM; ALCF caps user-accessible memory near 960 GB",
        "accel_vendor": "Intel",
        "accel_class": "gpu",
        "accel_model": "Intel Data Center GPU Max (Ponte Vecchio)",
        "accel_per_node": 6,
        "accel_note": "6 GPUs = 12 tiles; MPI ranks usually map to tiles",
        "accel_mem_gb": 768,
        "interconnect": "Slingshot 11, 8 endpoints per node, dragonfly",
        "node_storage": "None (use DAOS or Flare)",
        "facility_storage": "DAOS 230 PB @ 31 TB/s, Flare (Lustre), gecko-home",
        "scheduler": "PBS Pro",
        "login": "aurora.alcf.anl.gov",
        "filesystems": "flare",
        "queues": [
            ("debug", "small, short, good for first runs"),
            ("prod", "routing queue -> small / medium / large / backfill"),
        ],
        "jupyter": False,
        "jupyter_note": "No JupyterHub. Run Jupyter over an SSH tunnel to a compute node.",
        "granularity": "node",
        "max_nodes": 10624,
        "allocation": "INCITE, ALCC, Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 3, "inference": 2, "simulation": 3,
            "data": 2, "learning": 1, "novel_hw": 1, "viz": 2,
        },
        "watch_out": "Early-production instability is documented; checkpoint often. "
                     "The programming model is SYCL/oneAPI, not CUDA.",
        "software": "module load frameworks  (PyTorch, TensorFlow, oneCCL, Intel MPI)",
        "links": {
            "Getting started": f"{DOCS}/aurora/getting-started-on-aurora/",
            "Running jobs": f"{DOCS}/aurora/running-jobs-aurora/",
            "Known issues": f"{DOCS}/aurora/known-issues/",
            "System page": f"{WWW}/aurora",
        },
        "sources": [f"{WWW}/aurora", f"{DOCS}/aurora/"],
        "verified": True,
        "job": """#!/bin/bash -l
#PBS -N first_job
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare
#PBS -q debug
#PBS -A <your_project>

cd $PBS_O_WORKDIR
module load frameworks

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=12                       # one rank per GPU tile
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \\
        --depth=8 --cpu-bind depth \\
        python train.py""",
    },
    {
        "key": "polaris",
        "name": "Polaris",
        "family": "Supercomputer",
        "status": "production",
        "tagline": "NVIDIA A100 system. The default choice for CUDA and PyTorch work.",
        "vendor": "HPE Apollo 6500 Gen10+ / NVIDIA",
        "nodes": 560,
        "cpu": "1x AMD EPYC Milan 7543P @ 2.8 GHz",
        "cores_per_node": 32,
        "threads_per_node": 64,
        "cpu_mem_gb": 512,
        "cpu_mem_note": "512 GB DDR4",
        "accel_vendor": "NVIDIA",
        "accel_class": "gpu",
        "accel_model": "4x NVIDIA A100 40 GB, NVLink",
        "accel_per_node": 4,
        "accel_note": "MIG partitioning available in debug and debug-scaling",
        "accel_mem_gb": 160,
        "interconnect": "Slingshot 11, 2 adapters per node, dragonfly",
        "node_storage": "2x 1.6 TB NVMe in RAID0 (~3.2 TB scratch)",
        "facility_storage": "Eagle (Lustre), agile-home",
        "scheduler": "PBS Pro",
        "login": "polaris.alcf.anl.gov",
        "filesystems": "home:eagle",
        "queues": [
            ("debug", "8 dedicated nodes, up to 16-24 when free"),
            ("debug-scaling", "multi-node debugging"),
            ("prod", "routing queue -> small / medium / large / backfill"),
            ("preemptable", "can be killed without warning; use #PBS -r y"),
            ("demand", "by arrangement; preempts the preemptable queue"),
        ],
        "jupyter": True,
        "jupyter_note": "jupyter.alcf.anl.gov -> Login Polaris. Spawning starts a real batch job.",
        "granularity": "node",
        "max_nodes": 560,
        "allocation": "INCITE, ALCC, Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 3, "inference": 2, "simulation": 3,
            "data": 3, "learning": 3, "novel_hw": 0, "viz": 3,
        },
        "watch_out": "40 GB per GPU is the practical ceiling for model size. "
                     "The prod queue routes by node count, so a mismatched walltime is silently rejected.",
        "software": "module use /soft/modulefiles; module load conda; conda activate base",
        "links": {
            "Getting started": f"{DOCS}/polaris/getting-started/",
            "Running jobs": f"{DOCS}/polaris/running-jobs/",
            "Using GPUs / MIG": f"{DOCS}/polaris/running-jobs/using-gpus/",
            "JupyterHub": f"{DOCS}/services/jupyter-hub/",
            "System page": f"{WWW}/polaris",
        },
        "sources": [f"{WWW}/polaris", f"{DOCS}/polaris/"],
        "verified": True,
        "job": """#!/bin/bash -l
#PBS -N first_job
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A <your_project>

cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda
conda activate base

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4                        # one rank per A100
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \\
        --depth=8 --cpu-bind depth \\
        python train.py""",
    },
    {
        "key": "sophia",
        "name": "Sophia",
        "family": "Supercomputer",
        "status": "production",
        "tagline": "DGX A100 cluster. You can ask for a single GPU, which makes it the friendliest starting point.",
        "vendor": "NVIDIA DGX A100",
        "nodes": 24,
        "cpu": "2x AMD EPYC 7742 (Rome), 64 cores each",
        "cores_per_node": 128,
        "threads_per_node": 256,
        "cpu_mem_gb": 1024,
        "cpu_mem_note": "DGX A100 host memory",
        "accel_vendor": "NVIDIA",
        "accel_class": "gpu",
        "accel_model": "8x NVIDIA A100 (22 nodes @ 40 GB, 2 nodes @ 80 GB)",
        "accel_per_node": 8,
        "accel_note": "320 GB GPU memory per node; the two bigmem nodes have 640 GB",
        "accel_mem_gb": 320,
        "interconnect": "HDR200 InfiniBand, fat tree",
        "node_storage": "15 TB SSD, up to 25 Gb/s",
        "facility_storage": "Eagle (Lustre), agile-home",
        "scheduler": "PBS Pro",
        "login": "sophia.alcf.anl.gov",
        "filesystems": "home:eagle",
        "queues": [
            ("by-gpu", "individual GPUs - best for development and teaching"),
            ("by-node", "whole 8-GPU nodes, multi-node jobs"),
            ("bigmem", "the two 80 GB-per-GPU nodes, max 1 node"),
        ],
        "jupyter": True,
        "jupyter_note": "jupyter.alcf.anl.gov -> Login Sophia. Pick the by-gpu queue for a single GPU.",
        "granularity": "gpu",
        "max_nodes": 24,
        "allocation": "Director's Discretionary (also hosts the Inference Service)",
        "good_for": {
            "llm_api": 1, "train": 3, "inference": 3, "simulation": 1,
            "data": 3, "learning": 3, "novel_hw": 0, "viz": 2,
        },
        "watch_out": "Only 24 nodes total, so multi-node scaling studies belong on Polaris or Aurora. "
                     "Do not compile on the login nodes.",
        "software": "GNU compilers + CUDA on compute nodes; containers via Apptainer",
        "links": {
            "Getting started": f"{DOCS}/sophia/getting-started/",
            "Running jobs": f"{DOCS}/sophia/queueing-and-running-jobs/running-jobs/",
            "Fine-tuning with Autotrain": f"{DOCS}/sophia/data-science/fine-tune-LLM-with-Autotrain/",
            "JupyterHub": f"{DOCS}/services/jupyter-hub/",
        },
        "sources": [f"{DOCS}/sophia/", f"{DOCS}/sophia/queueing-and-running-jobs/running-jobs/"],
        "verified": True,
        "job": """#!/bin/bash -l
#PBS -N first_job
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -A <your_project>

cd $PBS_O_WORKDIR
nvidia-smi -L
python train.py

# Interactive single-GPU session instead:
# qsub -I -l select=1 -l walltime=1:00:00 -l filesystems=home:eagle \\
#      -q by-gpu -A <your_project>""",
    },
    {
        "key": "crux",
        "name": "Crux",
        "family": "Supercomputer",
        "status": "production",
        "tagline": "CPU-only cluster. Stop burning GPU hours on work that has no GPU in it.",
        "vendor": "HPE Cray EX, liquid cooled",
        "nodes": 256,
        "cpu": "2x AMD EPYC 7742 (Rome), 64 cores each",
        "cores_per_node": 128,
        "threads_per_node": 256,
        "cpu_mem_gb": 256,
        "cpu_mem_note": "128 GB DDR4 per socket",
        "accel_vendor": None,
        "accel_class": "none",
        "accel_model": "None",
        "accel_per_node": 0,
        "accel_note": "",
        "accel_mem_gb": 0,
        "interconnect": "Slingshot 11, 200 Gb",
        "node_storage": "None",
        "facility_storage": "Eagle (Lustre), agile-home",
        "scheduler": "PBS Pro",
        "login": "crux.alcf.anl.gov",
        "filesystems": "home:eagle",
        "queues": [
            ("workq-route", "routing queue -> workq"),
            ("preemptable", "can be killed without warning; use #PBS -r y"),
            ("demand", "by arrangement; preempts the preemptable queue"),
        ],
        "jupyter": False,
        "jupyter_note": "Not served by ALCF JupyterHub. Use SSH tunneling if you need a notebook.",
        "granularity": "node",
        "max_nodes": 256,
        "allocation": "Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 0, "inference": 0, "simulation": 3,
            "data": 3, "learning": 2, "novel_hw": 0, "viz": 1,
        },
        "watch_out": "About 1.18 PF total, so it is small next to Polaris and Aurora. "
                     "Set OMP_NUM_THREADS deliberately or you will oversubscribe 128 cores.",
        "software": "HPE Cray PE; module use /soft/modulefiles for ALCF-built software",
        "links": {
            "Getting started": f"{DOCS}/crux/getting-started/",
            "Running jobs": f"{DOCS}/crux/queueing-and-running-jobs/running-jobs/",
            "System page": f"{WWW}/crux",
        },
        "sources": [f"{DOCS}/crux/", f"{WWW}/crux"],
        "verified": True,
        "job": """#!/bin/bash -l
#PBS -N first_job
#PBS -l select=2:ncpus=128
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q workq-route
#PBS -A <your_project>

cd $PBS_O_WORKDIR

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=64
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
export OMP_NUM_THREADS=2

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \\
        --depth=2 --cpu-bind depth ./my_app""",
    },
    {
        "key": "inference",
        "name": "ALCF Inference Service",
        "family": "Service",
        "status": "production",
        "tagline": "OpenAI-compatible API over models hosted on Sophia and Metis. No allocation, no job script.",
        "vendor": "ALCF (Globus Compute + vLLM on Sophia; SambaNova SN40L for Metis)",
        "nodes": None,
        "cpu": "n/a - served for you",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "NVIDIA / SambaNova",
        "accel_class": "service",
        "accel_model": "Backed by Sophia A100s and Metis SN40L RDUs",
        "accel_per_node": 0,
        "accel_note": "",
        "accel_mem_gb": 0,
        "interconnect": "HTTPS",
        "node_storage": "n/a",
        "facility_storage": "n/a",
        "scheduler": "None - REST API",
        "login": None,
        "filesystems": "",
        "queues": [("n/a", "requests are queued by the service; check endpoint status via the API")],
        "jupyter": False,
        "jupyter_note": "Call it from any notebook, anywhere, including your laptop.",
        "granularity": "gpu",
        "max_nodes": 0,
        "allocation": "Open to all ALCF users. No allocation needed for Metis.",
        "good_for": {
            "llm_api": 3, "train": 0, "inference": 3, "simulation": 0,
            "data": 2, "learning": 3, "novel_hw": 0, "viz": 0,
        },
        "watch_out": "Metis supports chat completions only - no batch processing or tool calling. "
                     "Model availability changes; query the endpoint list rather than hardcoding names.",
        "software": "pip install openai globus_sdk",
        "links": {
            "Inference endpoints": f"{DOCS}/services/inference-endpoints/",
            "Metis endpoint guide": f"{DOCS}/ai-testbed/sn40l_inference/using_an_inference_endpoint/",
            "Overview": f"{WWW}/ai-inference",
        },
        "sources": [f"{DOCS}/services/inference-endpoints/",
                    f"{DOCS}/ai-testbed/sn40l_inference/using_an_inference_endpoint/"],
        "verified": True,
        "job": """# Runs from anywhere with internet - laptop, notebook, or an ALCF node.

pip install openai globus_sdk

wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
python inference_auth_token.py authenticate          # Globus login, once

access_token=$(python inference_auth_token.py get_access_token)

# What is running right now:
curl -X GET "https://inference-api.alcf.anl.gov/resource_server/list-endpoints" \\
     -H "Authorization: Bearer ${access_token}"

# A chat completion on Metis:
curl -X POST "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions" \\
     -H "Authorization: Bearer ${access_token}" \\
     -H "Content-Type: application/json" \\
     -d '{"model": "gpt-oss-120b",
          "messages": [{"role": "user", "content": "Hello"}]}'""",
    },
    {
        "key": "cerebras",
        "name": "Cerebras CS-3",
        "family": "AI Testbed",
        "status": "production",
        "tagline": "Wafer-scale training. One logical device instead of a cluster you have to shard by hand.",
        "vendor": "Cerebras",
        "nodes": 4,
        "cpu": "Cluster support nodes (worker, activation, MemoryX, SwarmX)",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "Host resources are managed by the appliance, not requested by you",
        "accel_vendor": "Cerebras",
        "accel_class": "novel",
        "accel_model": "4x CS-3 Wafer-Scale Engine, 900K cores each",
        "accel_per_node": 1,
        "accel_total": 4,
        "accel_note": "44 GB on-chip SRAM per WSE; weights stream from MemoryX",
        "accel_mem_gb": 44,
        "interconnect": "SwarmX fabric",
        "node_storage": "Shared /home, /projects, /software across the AI Testbed",
        "facility_storage": "AI Testbed NFS: 1 TB home, 2 TB project (default quotas)",
        "scheduler": "Wafer-Scale Cluster appliance (submit a job, it handles placement)",
        "login": "cerebras.alcf.anl.gov",
        "filesystems": "",
        "queues": [("appliance", "jobs are scheduled by the cluster, not by PBS")],
        "jupyter": False,
        "jupyter_note": "Port forwarding is documented if you need a browser tool.",
        "granularity": "node",
        "max_nodes": 4,
        "allocation": "Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 3, "inference": 1, "simulation": 0,
            "data": 0, "learning": 1, "novel_hw": 3, "viz": 0,
        },
        "watch_out": "Sized for models up to roughly 200B parameters. "
                     "Your PyTorch code needs porting to the Cerebras stack - budget time for it.",
        "software": "Cerebras PyTorch venv (path in Getting Started), then python run.py",
        "links": {
            "System overview": f"{DOCS}/ai-testbed/cerebras/",
            "Getting started": f"{DOCS}/ai-testbed/cerebras/getting-started/",
            "Running a model": f"{DOCS}/ai-testbed/cerebras/running-a-model-or-program/",
        },
        "sources": [f"{DOCS}/ai-testbed/cerebras/", f"{WWW}/alcf-ai-testbed"],
        "verified": True,
        "job": """ssh <ALCFUserID>@cerebras.alcf.anl.gov     # MobilePASS+ / CRYPTOCard passcode
ssh cer-usn-01                              # or cer-usn-02

# Activate the Cerebras PyTorch venv (exact path is in Getting Started),
# then launch through the appliance:
python run.py --mode train --params configs/my_model.yaml

# /home, /projects and /software are shared across all AI Testbed systems.""",
    },
    {
        "key": "sambanova",
        "name": "SambaNova DataScale (SN30)",
        "family": "AI Testbed",
        "status": "production",
        "tagline": "Reconfigurable dataflow units for large models that will not fit one GPU.",
        "vendor": "SambaNova",
        "nodes": 8,
        "cpu": "Host nodes sn30-r1-h1 ...",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "SambaNova",
        "accel_class": "novel",
        "accel_model": "8x Cardinal SN30 RDUs per node, 64 total (8 nodes in 4 racks)",
        "accel_per_node": 8,
        "accel_total": 64,
        "accel_note": "Programmed through SambaFlow, not CUDA. Each RDU pairs 640 MB "
                      "on-chip SRAM with ~1 TB of attached DDR - not comparable to GPU HBM "
                      "as a single number",
        "accel_mem_gb": 0,
        "interconnect": "SambaNova rack fabric",
        "node_storage": "Shared /home, /projects, /software across the AI Testbed",
        "facility_storage": "AI Testbed NFS: 1 TB home, 2 TB project (default quotas)",
        "scheduler": "SambaNova job launch on the host nodes",
        "login": "sambanova.alcf.anl.gov",
        "filesystems": "",
        "queues": [("host nodes", "connect to sm-01 / sm-02, then sn30-r1-hN")],
        "jupyter": False,
        "jupyter_note": "Port forwarding is documented for TensorBoard and similar tools.",
        "granularity": "node",
        "max_nodes": 8,
        "allocation": "Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 3, "inference": 2, "simulation": 0,
            "data": 0, "learning": 1, "novel_hw": 3, "viz": 0,
        },
        "watch_out": "This is a different system from Metis (SN40L). SN30 needs an allocation; "
                     "Metis is open to everyone through the Inference Service.",
        "software": "source /software/sambanova/envs/sn_env.sh",
        "links": {
            "AI Testbed overview": f"{DOCS}/ai-testbed/",
            "Data management": f"{DOCS}/ai-testbed/data-management/data-management-overview/",
            "Testbed system page": f"{WWW}/alcf-ai-testbed",
        },
        "sources": [f"{WWW}/alcf-ai-testbed", f"{DOCS}/ai-testbed/sambanova/"],
        "verified": True,
        "job": """ssh <ALCFUserID>@sambanova.alcf.anl.gov   # MobilePASS+ passcode
ssh sm-01                                  # or sm-02

source /software/sambanova/envs/sn_env.sh  # SambaFlow stack + venv

# Then run your SambaFlow model. Start from the examples in the user guide.""",
    },
    {
        "key": "graphcore",
        "name": "Graphcore Bow Pod64",
        "family": "AI Testbed",
        "status": "production",
        "tagline": "IPUs. Strong on sparse and graph-shaped models that map badly to GPUs.",
        "vendor": "Graphcore",
        "nodes": 4,
        "cpu": "Poplar host servers gc-poplar-02/03/04",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "Graphcore",
        "accel_class": "novel",
        "accel_model": "64 Bow IPUs, about 22 PF/s (FP16)",
        "accel_per_node": 0,
        "accel_total": 64,
        "accel_note": "57.6 GB total In-Processor-Memory across 94,208 IPU cores. IPUs "
                      "are assigned by SLURM, not pinned to a host, so there is no "
                      "meaningful per-node count",
        "accel_mem_gb": 0,
        "interconnect": "IPU-Fabric",
        "node_storage": "Shared /home, /projects, /software across the AI Testbed",
        "facility_storage": "AI Testbed NFS: 1 TB home, 2 TB project (default quotas)",
        "scheduler": "SLURM",
        "login": None,
        "filesystems": "",
        "queues": [("SLURM partitions", "srun / sbatch; check with sinfo")],
        "jupyter": False,
        "jupyter_note": "",
        "granularity": "node",
        "max_nodes": 4,
        "allocation": "Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 2, "inference": 2, "simulation": 0,
            "data": 0, "learning": 1, "novel_hw": 3, "viz": 0,
        },
        "watch_out": "The Poplar SDK is its own toolchain. This is the only ALCF system on SLURM, "
                     "so your PBS habits will not transfer.",
        "software": "Poplar SDK; PopTorch for PyTorch models",
        "links": {
            "Getting started": f"{DOCS}/ai-testbed/graphcore/getting-started/",
            "Running a model": f"{DOCS}/ai-testbed/graphcore/running-a-model-or-program/",
            "Job queuing": f"{DOCS}/ai-testbed/graphcore/job-queuing-and-submission/",
        },
        "sources": [f"{DOCS}/ai-testbed/graphcore/getting-started/", f"{WWW}/alcf-ai-testbed"],
        "verified": True,
        "job": """# Log in to the Graphcore login node (hostname is in Getting Started),
# then hop to a Poplar host:
ssh gc-poplar-02.ai.alcf.anl.gov            # or -03, -04

# Graphcore is the SLURM system at ALCF:
sinfo
srun --ipus=4 python my_poptorch_model.py""",
    },
    {
        "key": "groq",
        "name": "GroqRack",
        "family": "AI Testbed",
        "status": "production",
        "tagline": "LPUs built for deterministic, very low latency inference.",
        "vendor": "Groq",
        "nodes": 9,
        "cpu": "groq-r01-gn-01 ... gn-09 host nodes",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "Groq",
        "accel_class": "novel",
        "accel_model": "8 LPU cards per node, 72 per rack",
        "accel_per_node": 8,
        "accel_total": 72,
        "accel_note": "Larger models can require the whole 9-node rack",
        "accel_mem_gb": 0,
        "interconnect": "Groq rack fabric",
        "node_storage": "Shared /home, /projects, /software across the AI Testbed",
        "facility_storage": "AI Testbed NFS: 1 TB home, 2 TB project (default quotas)",
        "scheduler": "PBS Pro",
        "login": "groq-login-01.ai.alcf.anl.gov",
        "filesystems": "",
        "queues": [("rack nodes", "qstat -wa before you start; some models need all 9 nodes")],
        "jupyter": False,
        "jupyter_note": "GroqView profiling runs over an SSH tunnel.",
        "granularity": "node",
        "max_nodes": 9,
        "allocation": "Director's Discretionary",
        "good_for": {
            "llm_api": 0, "train": 0, "inference": 3, "simulation": 0,
            "data": 0, "learning": 1, "novel_hw": 3, "viz": 0,
        },
        "watch_out": "Inference only - there is no training path. "
                     "Check for other users before taking the rack.",
        "software": "GroqFlow; groqit() compiles your model ahead of time",
        "links": {
            "Getting started": f"{DOCS}/ai-testbed/groq/getting-started/",
            "Examples": f"{DOCS}/ai-testbed/groq/examples/",
            "GroqView profiling": f"{DOCS}/ai-testbed/groq/groqview/",
        },
        "sources": [f"{DOCS}/ai-testbed/groq/getting-started/", f"{DOCS}/ai-testbed/groq/examples/"],
        "verified": True,
        "job": """ssh <ALCFUserID>@groq-login-01.ai.alcf.anl.gov
ssh groq-r01-gn-01.ai.alcf.anl.gov

qstat -wa                                   # is anyone else on the rack?

# Compile ahead of time with GroqFlow, then run:
python my_groqflow_script.py""",
    },
    {
        "key": "tara",
        "name": "Tara",
        "family": "AI inference system",
        "status": "coming online",
        "tagline": "672 Grace Hopper nodes dedicated to inference. Coming online in 2026.",
        "vendor": "HPE EX254n / NVIDIA",
        "nodes": 672,
        "cpu": "4x NVIDIA Grace CPU per node (2,688 total)",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 480,
        "cpu_mem_note": "480 GB per node, 322 TB system-wide",
        "accel_vendor": "NVIDIA",
        "accel_class": "gpu",
        "accel_model": "4x NVIDIA GH200 Grace Hopper Superchip (2,688 total)",
        "accel_per_node": 4,
        "accel_total": 2688,
        "accel_note": "2.66 EF/s theoretical FP16/BF16",
        "accel_mem_gb": 384,
        "interconnect": "Slingshot-200",
        "node_storage": "TBD",
        "facility_storage": "TBD",
        "scheduler": "TBD",
        "login": None,
        "filesystems": "",
        "queues": [("TBD", "not yet in production")],
        "jupyter": False,
        "jupyter_note": "",
        "granularity": "node",
        "max_nodes": 672,
        "allocation": "Will support the ALCF Inference Service",
        "good_for": {
            "llm_api": 2, "train": 1, "inference": 3, "simulation": 0,
            "data": 1, "learning": 0, "novel_hw": 1, "viz": 0,
        },
        "watch_out": "Announced, not yet available to users. Plan on Sophia or Metis today.",
        "software": "TBD",
        "links": {"System page": f"{WWW}/tara"},
        "sources": [f"{WWW}/tara"],
        "verified": True,
        "job": "",
    },
    {
        "key": "minerva",
        "name": "Minerva",
        "family": "AI inference system",
        "status": "coming online",
        "tagline": "64 NVIDIA B200s backing the ALCF Inference Service.",
        "vendor": "World Wide Technology / NVIDIA",
        "nodes": None,
        "cpu": "Not published",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "NVIDIA",
        "accel_class": "gpu",
        "accel_model": "64x NVIDIA B200",
        "accel_per_node": 0,
        "accel_total": 64,
        "accel_note": "Per-node configuration not published yet",
        "accel_mem_gb": 0,
        "interconnect": "NVIDIA Quantum-2 InfiniBand",
        "node_storage": "Not published",
        "facility_storage": "Not published",
        "scheduler": "Not published",
        "login": None,
        "filesystems": "",
        "queues": [("n/a", "reached through the Inference Service, not by SSH")],
        "jupyter": False,
        "jupyter_note": "",
        "granularity": "gpu",
        "max_nodes": 0,
        "allocation": "Supports the ALCF Inference Service",
        "good_for": {
            "llm_api": 2, "train": 0, "inference": 3, "simulation": 0,
            "data": 0, "learning": 0, "novel_hw": 0, "viz": 0,
        },
        "watch_out": "Only the GPU count is public. Use the Inference Service rather than "
                     "planning a direct-access workflow.",
        "software": "n/a",
        "links": {"System page": f"{WWW}/minerva"},
        "sources": [f"{WWW}/minerva"],
        "verified": False,
        "job": "",
    },
    {
        "key": "janus",
        "name": "Janus",
        "family": "AI inference system",
        "status": "coming online",
        "tagline": "64 NVIDIA H100s aimed at workforce training and research.",
        "vendor": "HPE / NVIDIA",
        "nodes": None,
        "cpu": "Not published",
        "cores_per_node": 0,
        "threads_per_node": 0,
        "cpu_mem_gb": 0,
        "cpu_mem_note": "",
        "accel_vendor": "NVIDIA",
        "accel_class": "gpu",
        "accel_model": "64x NVIDIA H100",
        "accel_per_node": 0,
        "accel_total": 64,
        "accel_note": "Per-node configuration not published yet",
        "accel_mem_gb": 0,
        "interconnect": "Not published",
        "node_storage": "Not published",
        "facility_storage": "Not published",
        "scheduler": "Not published",
        "login": None,
        "filesystems": "",
        "queues": [("n/a", "not yet open to general users")],
        "jupyter": False,
        "jupyter_note": "",
        "granularity": "node",
        "max_nodes": 0,
        "allocation": "Positioned for training and workforce development",
        "good_for": {
            "llm_api": 0, "train": 2, "inference": 2, "simulation": 0,
            "data": 0, "learning": 3, "novel_hw": 0, "viz": 0,
        },
        "watch_out": "Announced for hands-on AI and HPC training. Details are not published yet - "
                     "if you are planning a class now, use Polaris or Sophia.",
        "software": "n/a",
        "links": {"System page": f"{WWW}/janus"},
        "sources": [f"{WWW}/janus"],
        "verified": False,
        "job": "",
    },
]

ACCESS_LINKS = {
    "Request a Director's Discretionary project": f"{WWW}/science/directors-discretionary-allocation-program",
    "INCITE program": f"{WWW}/science/incite-allocation-program",
    "ALCC program": f"{WWW}/science/alcc-allocation-program",
    "Onboarding your award": f"{WWW}/onboarding-your-project",
    "Get your token (CRYPTOCard / MobilePASS+)": f"{DOCS}/account-project-management/accounts-and-access/obtaining-a-token/",
    "MyALCF portal": "https://my.alcf.anl.gov",
    "Machine status": f"{WWW}/support-center/machine-status",
    "AskALCF chatbot": "https://ask.alcf.anl.gov",
    "Submit a support ticket": f"{DOCS}/support/ticket/",
}

# ---------------------------------------------------------------------------
# New-user material: plain-English glossary + first-week checklist
# ---------------------------------------------------------------------------
GLOSSARY = {
    "Node": "One physical computer inside the cluster. A node has CPUs, memory, and "
            "(on most systems) accelerators. Most ALCF machines are rented out one whole "
            "node at a time; Sophia and the Inference Service are the exceptions.",
    "Accelerator / GPU": "The chip that does the heavy math - NVIDIA or Intel GPUs, or novel "
            "AI chips (Cerebras wafers, SambaNova RDUs, Graphcore IPUs, Groq LPUs). "
            "'Accelerator memory' is the fast memory attached to that chip; if your model or "
            "data does not fit in it, you shard across more of them.",
    "Core vs thread": "A core is a physical execution unit in the CPU; a thread is a logical "
            "one (usually 2 per core). Job scripts and performance planning use physical cores. "
            "This app lists both so you do not accidentally double-count.",
    "Login node vs compute node": "SSH lands you on a login node - a shared machine for "
            "editing, compiling, and submitting jobs. Actual work runs on compute nodes, which "
            "the scheduler assigns to you. Never run heavy jobs on the login node; everyone is on it.",
    "Scheduler (PBS Pro / SLURM)": "The system that queues and launches jobs. Almost all of "
            "ALCF uses PBS Pro (commands: qsub, qstat, qdel). Graphcore uses SLURM "
            "(sbatch, squeue, scancel). You describe what you need; it finds the hardware.",
    "Queue": "A named lane inside the scheduler with its own limits and priorities. 'debug' "
            "queues are short and small - made for first runs. 'prod' queues are for real "
            "workloads. 'preemptable' jobs are cheaper but can be killed at any time.",
    "Job script": "A shell script whose #PBS (or #SBATCH) header lines tell the scheduler what "
            "you need: node count, walltime, queue, project. The 'Start here' tab generates "
            "one for you.",
    "Walltime": "The wall-clock time limit you request for a job. When it expires the job is "
            "killed, finished or not - so checkpoint long runs. Shorter walltimes usually "
            "schedule faster.",
    "Allocation / project": "Your budget of compute hours, attached to a project with a PI. "
            "Every job charges hours to a project (#PBS -A <project>). No project, no jobs - "
            "except the Inference Service, which needs no allocation.",
    "INCITE / ALCC / Director's Discretionary": "The three allocation programs. INCITE and "
            "ALCC are big annual awards for established campaigns. Director's Discretionary "
            "(DD) is the year-round starting point - smaller, faster to get, fine for new work.",
    "MPI / rank": "MPI is the message-passing library that lets one program run across many "
            "nodes. Each cooperating process is a 'rank'. On GPUs, a common pattern is one "
            "rank per GPU (or per GPU tile on Aurora).",
    "Parallel file system (Eagle / Flare / DAOS)": "Big shared storage that every node can "
            "read and write. Eagle serves Polaris, Sophia and Crux; Flare and DAOS serve "
            "Aurora. Node-local scratch (NVMe/SSD) is separate: faster, but it vanishes when "
            "the job ends.",
    "Globus": "The service used to move data in and out of ALCF - think of it as a managed, "
            "restartable file transfer between institutions. Also the login mechanism for the "
            "Inference Service.",
    "JupyterHub": "jupyter.alcf.anl.gov gives you notebooks on Polaris and Sophia; spawning a "
            "notebook starts a real batch job behind the scenes, so it charges your project.",
    "MobilePASS+ / CRYPTOCard": "Your login token. The SSH password is the one-time passcode "
            "it generates (physical token: PIN + eight digits; app: enter PIN, type the code "
            "shown). There is no permanent password.",
    "Module system": "How software is provided on the clusters. 'module load frameworks' or "
            "'module load conda' puts compilers and Python stacks on your path instead of you "
            "installing them.",
    "AI Testbed": "ALCF's collection of non-GPU AI machines (Cerebras, SambaNova, Graphcore, "
            "Groq). Great for benchmarking and porting, but each has its own toolchain - "
            "budget porting time.",
    "Interactive job": "A scheduler job that gives you a live shell on a compute node "
            "(qsub -I ...). The right way to poke at hardware, debug, or run something by "
            "hand without abusing the login node.",
}

CHECKLIST = [
    ("Get on a project",
     "Ask your PI to add you to an existing ALCF project, or request a Director's "
     "Discretionary allocation (open year-round). Everything else waits on this."),
    ("Request your ALCF account",
     "Once the project exists, request an account at my.alcf.anl.gov and join the project. "
     "Approval involves a vetting step, so start early."),
    ("Set up your login token",
     "Enroll a MobilePASS+ soft token (or receive a CRYPTOCard). Do one test login: "
     "ssh <you>@polaris.alcf.anl.gov with the one-time passcode."),
    ("Pick your system with this app",
     "Set the task, scale, and hard requirements in the sidebar and read the Best match "
     "and Compare tabs. When in doubt between Polaris and Sophia, start on Sophia's "
     "by-gpu queue - it is the friendliest on-ramp."),
    ("Load the software environment",
     "Each system has a one-liner (shown under Full specs -> Software environment). "
     "Run it on the login node and confirm python / your compiler resolves."),
    ("Move your data with Globus",
     "Transfer code and datasets to your project directory on the right file system "
     "(Eagle for Polaris/Sophia/Crux, Flare for Aurora). Do not scp large data."),
    ("Run a debug job",
     "Download the job script from the Start here tab, replace <your_project>, and submit "
     "to the debug (or by-gpu) queue. Watch it with qstat -u $USER."),
    ("Check machine status and get help",
     "Bookmark the machine status page and the docs. AskALCF (ask.alcf.anl.gov) answers "
     "queue/module questions instantly; support@alcf.anl.gov for the rest."),
]

# ---------------------------------------------------------------------------
# Quick-start presets: one pick fills in every sidebar widget. New users
# rarely know what to select - these encode the most common asks.
# Keys here are the widget keys used in the sidebar below.
# ---------------------------------------------------------------------------
CUSTOM_SCENARIO = "Set it yourself below"
PRESETS = {
    "Fine-tune or train a model on NVIDIA GPUs": {
        "task_sel": "Train or fine-tune a model",
        "scale_sel": "2-10 nodes",
        "accel_sel": "NVIDIA (CUDA)",
    },
    "Use an LLM through an API right now": {
        "task_sel": "Use an LLM through an API (no setup)",
        "scale_sel": "Part of one accelerator",
        "accel_sel": "No preference",
    },
    "Large MPI simulation (CFD / MD / QCD)": {
        "task_sel": "Large-scale simulation (MPI / CFD / MD / QCD)",
        "scale_sel": "10-100 nodes",
        "accel_sel": "No preference",
    },
    "CPU data crunching, no GPUs needed": {
        "task_sel": "Data analysis, preprocessing, workflows",
        "scale_sel": "One node",
        "accel_sel": "CPU only",
    },
    "Brand new: learn HPC with notebooks": {
        "task_sel": "Learn HPC, teach a class, run notebooks",
        "scale_sel": "Part of one accelerator",
        "accel_sel": "No preference",
        "nb_chk": True,
    },
    "Port a model to novel AI hardware": {
        "task_sel": "Benchmark or port code to novel AI hardware",
        "scale_sel": "One node",
        "accel_sel": "Non-GPU AI accelerator",
    },
}


def apply_preset():
    """on_change callback: copy the chosen preset into the widget state.
    Hard-requirement sliders reset to zero so a stale minimum from earlier
    fiddling cannot silently distort a fresh scenario."""
    values = PRESETS.get(st.session_state.get("preset_sel"))
    if not values:
        return
    reset = {"min_accel_sl": 0, "min_accel_mem_sl": 0, "min_cores_sl": 0,
             "min_cpu_mem_sl": 0, "nb_chk": False}
    for key, val in {**reset, **values}.items():
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Line-by-line job script guides, so the generated script is not a magic
# incantation. Generic on purpose: they explain the pattern, not one file.
# ---------------------------------------------------------------------------
PBS_LINE_GUIDE = [
    ("#!/bin/bash -l", "run as a login shell so `module` commands work inside the job"),
    ("#PBS -N first_job", "the job's name - what you will see in `qstat`"),
    ("#PBS -l select=N", "how many nodes you are asking for"),
    ("#PBS -l walltime=00:30:00", "time limit (hh:mm:ss); the job is killed when it "
     "expires, finished or not - checkpoint anything long"),
    ("#PBS -l filesystems=...", "which shared file systems the job needs; if one is "
     "down, the job waits instead of failing mid-run"),
    ("#PBS -q debug", "which queue to wait in - debug is small, short, and schedules fast"),
    ("#PBS -A <your_project>", "the project whose allocation pays the hours; replace "
     "the placeholder or the job is rejected"),
    ("cd $PBS_O_WORKDIR", "start in the directory you ran `qsub` from - "
     "otherwise jobs start in $HOME"),
    ("wc -l < $PBS_NODEFILE", "$PBS_NODEFILE lists the nodes you were given; "
     "counting its lines gives your node count"),
    ("NRANKS_PER_NODE", "MPI processes per node; the convention is one rank per GPU "
     "(one per GPU *tile* on Aurora)"),
    ("mpiexec -n T --ppn P", "launch T processes in total, P on each node"),
    ("--depth=8 --cpu-bind depth", "give each rank 8 CPU cores and pin it to them, "
     "so ranks do not fight over the same cores"),
    ("qsub first_job.sh", "submit the script; then `qstat -u $USER` shows your place "
     "in the queue and `qdel <id>` cancels"),
]

SLURM_LINE_GUIDE = [
    ("sinfo", "show SLURM partitions and how many hosts are idle"),
    ("srun --ipus=4 python my_model.py", "run the command with 4 IPUs allocated by "
     "SLURM; sbatch submits a script instead of running interactively"),
    ("squeue / scancel", "SLURM's equivalents of `qstat` and `qdel`"),
]

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def evaluate(system, req):
    """Return (fit_score, misses, reasons, breakdown).

    misses    = hard requirements the system fails. Anything with misses is
                demoted below every system that meets the requirements.
    reasons   = short, human phrases explaining the fit (shown as pills).
    breakdown = (points, explanation) pairs; the score is exactly their sum.
                Shown in the UI so the ranking is never a black box.
    """
    misses, reasons, breakdown = [], [], []
    cls = system["accel_class"]
    is_service = cls == "service"

    def check_min(value, wanted, unit, label):
        """A hard minimum. A value of 0 means 'not published', which is a miss
        rather than a silent pass - unknown specs must never satisfy a
        requirement just because the number is absent."""
        if wanted <= 0:
            return
        if not value:
            misses.append(f"{label} not published")
        elif value < wanted:
            misses.append(f"only {value} {unit}")

    # --- Hard requirements -------------------------------------------------
    want = req["accel"]
    if want == "NVIDIA (CUDA)" and system["accel_vendor"] != "NVIDIA":
        misses.append("not an NVIDIA system")
    elif want == "Intel (SYCL / oneAPI)" and system["accel_vendor"] != "Intel":
        misses.append("not an Intel GPU system")
    elif want == "Non-GPU AI accelerator" and cls != "novel":
        misses.append("not a novel AI accelerator")
    elif want == "CPU only" and cls != "none":
        misses.append("has accelerators you said you do not need")

    if req["min_accel"] > 0 and cls in ("none", "service"):
        misses.append("no accelerators to request per node")
    else:
        check_min(system["accel_per_node"], req["min_accel"],
                  "accelerators per node", "accelerators per node")

    if is_service:
        # Per-node CPU and memory are the service's problem, not yours.
        if req["min_cores"] > 0 or req["min_cpu_mem"] > 0 or req["min_accel_mem"] > 0:
            reasons.append("hardware is managed for you")
    else:
        check_min(system["cores_per_node"], req["min_cores"],
                  "cores per node", "core count")
        check_min(system["cpu_mem_gb"], req["min_cpu_mem"],
                  "GB CPU memory per node", "CPU memory")
        check_min(system["accel_mem_gb"], req["min_accel_mem"],
                  "GB accelerator memory per node", "accelerator memory")

    if req["notebooks"] and not system["jupyter"] and not is_service:
        misses.append("no ALCF JupyterHub")

    if req["available_now"] and system["status"] != "production":
        misses.append("not available to users yet")

    needed = SCALES[req["scale"]]
    if not is_service and needed >= 1:
        # An unpublished node count must not silently satisfy a scale
        # requirement (Minerva and Janus publish no per-node structure).
        if not system["max_nodes"]:
            misses.append("node count not published")
        elif needed > system["max_nodes"]:
            misses.append(f"tops out at {system['max_nodes']} nodes")

    # --- Fit ---------------------------------------------------------------
    # Every point is accounted for in `breakdown` so the UI can show its work.
    task_fit = system["good_for"].get(req["task"], 0)
    fit_label = {
        3: "built for this kind of work",
        2: "handles this well",
        1: "possible, but not its strength",
    }.get(task_fit)
    if fit_label:
        breakdown.append((task_fit * 30, f"task fit: {fit_label}"))
        reasons.append(fit_label)
    else:
        breakdown.append((0, "task fit: wrong tool for this task"))

    if needed == 0 and system["granularity"] == "gpu":
        breakdown.append((20, "lets you request a single accelerator"))
        reasons.append("you can request a single accelerator")
    if needed >= 100 and system["max_nodes"] >= 500:
        breakdown.append((20, "scales to the node count you asked for"))
        reasons.append("scales to the node count you asked for")
    if req["notebooks"] and system["jupyter"]:
        breakdown.append((10, "served by ALCF JupyterHub"))
        reasons.append("served by ALCF JupyterHub")

    if system["status"] == "production":
        breakdown.append((10, "in production today"))
    if not system["verified"]:
        breakdown.append((-10, "specs only partly published"))
        reasons.append("specs only partly published")

    score = sum(points for points, _ in breakdown)
    return score, misses, reasons, breakdown


def accel_total(system):
    """Fleet-wide accelerator count, explicit where published and derived where
    node structure makes it meaningful."""
    if system.get("accel_total"):
        return system["accel_total"]
    if system["accel_per_node"] and system["nodes"]:
        return system["accel_per_node"] * system["nodes"]
    return None


def spec_row(system):
    total = accel_total(system)
    return {
        "System": system["name"],
        "Family": system["family"],
        "Status": system["status"],
        # Missing numerics are None (not "-") so every numeric column keeps a
        # clean nullable-integer dtype - mixing strings and ints breaks Arrow
        # serialization and forfeits numeric sorting in the table.
        "Nodes": system["nodes"] if system["nodes"] else None,
        "Accelerators/node": system["accel_per_node"] or None,
        "Accelerators total": f"{total:,}" if total else "-",
        "Accelerator": system["accel_model"],
        "Cores/node": system["cores_per_node"] or None,
        "CPU mem/node (GB)": system["cpu_mem_gb"] or None,
        "Accel mem/node (GB)": system["accel_mem_gb"] or None,
        "Scheduler": system["scheduler"],
        "Login": system["login"] or "-",
    }


def matches_query(system, query):
    """Case-insensitive free-text search across the fields a person would
    actually search by."""
    hay = " ".join([
        system["name"], system["family"], system["status"], system["tagline"],
        system["vendor"], system["cpu"], system["accel_model"],
        system["scheduler"], system["allocation"],
        system["accel_vendor"] or "",
    ]).lower()
    return all(word in hay for word in query.lower().split())


# ---------------------------------------------------------------------------
# Styling - deliberately small. Streamlit's own class names change between
# releases, so everything here targets classes this file defines.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Theme-neutral colors only: Streamlit renders in light OR dark mode, so
       plain text must inherit the theme color or use mid-grays readable on
       both. Elements with their own background (specstrip, pills) may set
       explicit foregrounds. */
    .eyebrow {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: #8494a4; margin-bottom: 0.35rem;
    }
    .masthead h1 {
        font-size: 2.1rem; font-weight: 700; margin: 0 0 0.25rem 0; color: inherit;
    }
    .masthead p { color: #8494a4; margin: 0; max-width: 62ch; }
    .rule { border-top: 2px solid rgba(132, 148, 164, 0.55); margin: 0.9rem 0 1.4rem 0; }
    .specstrip {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem; color: #23394b; background: #f2f6f9;
        border-left: 3px solid #0b7f8c; padding: 0.55rem 0.75rem;
        border-radius: 0 4px 4px 0; margin: 0.4rem 0 0.7rem 0; line-height: 1.6;
    }
    .pill {
        display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem; padding: 0.12rem 0.5rem; border-radius: 3px;
        margin-right: 0.3rem; margin-bottom: 0.25rem;
    }
    .pill-ok   { background: #dff0e6; color: #14572f; }
    .pill-miss { background: #fbe3e3; color: #7d1d1d; }
    .pill-soon { background: #fdf0d5; color: #7a5300; }
    .caveat { color: #8494a4; font-size: 0.85rem; }
    .scoreline {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.85rem; color: inherit; opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="masthead">
      <div class="eyebrow">Argonne Leadership Computing Facility</div>
      <h1>Which system should I use?</h1>
      <p>Describe the work you need done. This ranks ALCF's systems against it and
         hands you the exact commands for your first job.</p>
    </div>
    <div class="rule"></div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Quick start</div>', unsafe_allow_html=True)
    st.selectbox(
        "Common scenarios",
        [CUSTOM_SCENARIO] + list(PRESETS),
        key="preset_sel",
        on_change=apply_preset,
        help="Picking one fills in everything below. You can still adjust any "
             "field afterwards - this is a starting point, not a lock.",
    )

    st.markdown('<div class="eyebrow">Your work</div>', unsafe_allow_html=True)

    task_label = st.selectbox(
        "What are you doing?", list(TASKS.keys()), key="task_sel",
        help="The closest match is fine - this drives the ranking more than "
             "anything else.",
    )
    scale = st.selectbox(
        "How much do you need at once?",
        list(SCALES.keys()),
        index=1,
        key="scale_sel",
        help="Node counts, not total hours. Pick the largest single job you expect "
             "to run. A node is one physical machine - see the New here? tab.",
    )
    accel = st.selectbox(
        "Hardware you need", ACCEL_CHOICES, key="accel_sel",
        help="Only pick a vendor if your code requires it (CUDA-only, SYCL, ...). "
             "'No preference' keeps every option open.",
    )

    st.markdown('<div class="eyebrow">Hard requirements</div>', unsafe_allow_html=True)
    st.caption("Leave at zero unless your code genuinely requires it.")

    min_accel = st.slider(
        "Accelerators per node, minimum", 0, 8, 0, key="min_accel_sl",
        help="Accelerator = GPU or AI chip. 'Per node' = inside one physical "
             "machine. Jargon is decoded in the New here? tab.",
    )
    min_accel_mem = st.slider(
        "Accelerator memory per node (GB), minimum", 0, 800, 0, step=40,
        key="min_accel_mem_sl",
        help="All accelerators in one node combined - not per GPU. Rule of thumb: "
             "your model, activations, and optimizer state must fit here.",
    )
    min_cores = st.slider(
        "Physical CPU cores per node, minimum", 0, 128, 0, step=8,
        key="min_cores_sl",
        help="Physical cores, not hyperthreads - marketing numbers often double "
             "them. The Full specs tab lists both.",
    )
    min_cpu_mem = st.slider(
        "CPU memory per node (GB), minimum", 0, 1024, 0, step=64,
        key="min_cpu_mem_sl",
        help="Ordinary RAM on the node, separate from accelerator memory. Matters "
             "for big preprocessing and data loading.",
    )

    notebooks = st.checkbox(
        "I need JupyterHub notebooks", key="nb_chk",
        help="Jupyter in the browser at jupyter.alcf.anl.gov - served for Polaris "
             "and Sophia only. Spawning a notebook starts a real batch job.",
    )
    available_now = st.checkbox(
        "Only systems I can use today", value=True, key="avail_chk",
        help="Untick to include announced systems (Tara, Minerva, Janus) in the "
             "ranking - useful for planning, useless for running jobs now.",
    )
    hide_misses = st.checkbox(
        "Hide systems that miss a requirement", value=False, key="hide_chk",
        help="Misses already demote a system below every clean match; this "
             "removes them from the list entirely.",
    )

req = {
    "task": TASKS[task_label],
    "scale": scale,
    "accel": accel,
    "min_accel": min_accel,
    "min_accel_mem": min_accel_mem,
    "min_cores": min_cores,
    "min_cpu_mem": min_cpu_mem,
    "notebooks": notebooks,
    "available_now": available_now,
}

results = []
for system in SYSTEMS:
    score, misses, reasons, breakdown = evaluate(system, req)
    results.append({
        "sys": system, "score": score, "misses": misses,
        "reasons": reasons, "breakdown": breakdown,
    })

results.sort(key=lambda r: (len(r["misses"]) == 0, r["score"]), reverse=True)
clean = [r for r in results if not r["misses"]]
by_name = {r["sys"]["name"]: r for r in results}

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
if not clean:
    st.warning(
        "Nothing at ALCF meets every requirement as stated. The closest match is below - "
        "loosen a slider, or email support@alcf.anl.gov and describe the workload."
    )

top = results[0]
sysname = top["sys"]["name"]

st.markdown('<div class="eyebrow">Best match</div>', unsafe_allow_html=True)
st.subheader(sysname)
st.caption(f"Ranked #1 for: {task_label}  ·  {scale}  ·  {accel}")
st.write(top["sys"]["tagline"])

pills = "".join(f'<span class="pill pill-ok">{r}</span>' for r in top["reasons"][:4])
pills += "".join(f'<span class="pill pill-miss">{m}</span>' for m in top["misses"][:3])
if top["sys"]["status"] != "production":
    pills += '<span class="pill pill-soon">not in production yet</span>'
st.markdown(pills, unsafe_allow_html=True)

runners = (clean if clean else results)[1:3]
if runners:
    st.markdown(
        "**Also consider:** " + "  ·  ".join(
            f"{r['sys']['name']} ({r['score']} pts)" for r in runners
        ) + "  -  full comparison in the *Compare* tab."
    )


def render_specstrip(s):
    bits = []
    if s["nodes"]:
        bits.append(f"nodes {s['nodes']:,}")
    if s["accel_per_node"]:
        bits.append(f"accel/node {s['accel_per_node']}")
    elif accel_total(s):
        bits.append(f"accel total {accel_total(s):,}")
    if s["cores_per_node"]:
        bits.append(f"cores/node {s['cores_per_node']}")
    if s["cpu_mem_gb"]:
        bits.append(f"cpu mem {s['cpu_mem_gb']} GB")
    if s["accel_mem_gb"]:
        bits.append(f"accel mem {s['accel_mem_gb']} GB")
    bits.append(f"sched {s['scheduler'].split('(')[0].strip()}")
    st.markdown(f'<div class="specstrip">{"  &middot;  ".join(bits)}</div>', unsafe_allow_html=True)


def render_breakdown(r):
    """The full arithmetic behind a system's score. No black boxes."""
    lines = [f'<span class="scoreline">{pts:+d}&nbsp;&nbsp;{label}</span>'
             for pts, label in r["breakdown"]]
    st.markdown("<br>".join(lines), unsafe_allow_html=True)
    st.markdown(
        f'<div class="scoreline"><b>= {r["score"]} points</b></div>',
        unsafe_allow_html=True,
    )
    if r["misses"]:
        st.markdown(
            "**Hard requirements missed:** " + ", ".join(r["misses"]) +
            ". A system with any miss ranks below every system with none, "
            "regardless of score."
        )


render_specstrip(top["sys"])

with st.expander(f"How this was scored - {top['score']} points"):
    render_breakdown(top)
    st.caption(
        "Task fit is worth up to +90 (it dominates on purpose - the right kind of machine "
        "beats a bigger wrong one). Single-accelerator access and large-scale headroom are "
        "worth +20 each when your scale calls for them, JupyterHub +10 when you asked for "
        "notebooks, production status +10, and partly-published specs cost -10."
    )

tab_start, tab_specs, tab_compare, tab_all, tab_new, tab_access = st.tabs(
    ["Start here", "Full specs", "Compare", "All systems", "New here?", "Getting an account"]
)

# --- Start here ------------------------------------------------------------
with tab_start:
    s = top["sys"]
    if s["status"] != "production":
        st.info(f"{s['name']} is announced but not open to users yet. "
                "The steps below will change once it enters production.")

    if s["login"]:
        st.markdown('<div class="eyebrow">1. Connect</div>', unsafe_allow_html=True)
        st.code(f"ssh <your_username>@{s['login']}", language="bash")
        st.caption(
            "The password is the passcode from your CRYPTOCard or MobilePASS+ token. "
            "Physical token: PIN followed by the eight digits. Mobile: enter your PIN in "
            "the app, then type the passcode it shows."
        )
    else:
        st.markdown('<div class="eyebrow">1. Connect</div>', unsafe_allow_html=True)
        st.caption("No direct SSH host published for this system - see the links under Full specs.")

    if s["job"]:
        st.markdown('<div class="eyebrow">2. Your first job</div>', unsafe_allow_html=True)
        st.code(s["job"], language="bash")
        if s["scheduler"].startswith("PBS"):
            st.code(f"qsub first_job.sh\nqstat -u $USER", language="bash")
        st.download_button(
            f"Download this as first_job_{s['key']}.sh",
            s["job"],
            file_name=f"first_job_{s['key']}.sh",
        )

        line_guide = None
        if s["scheduler"].startswith("PBS"):
            line_guide = PBS_LINE_GUIDE
        elif s["scheduler"] == "SLURM":
            line_guide = SLURM_LINE_GUIDE
        if line_guide:
            with st.expander("New to job scripts? What every line means"):
                for fragment, why in line_guide:
                    st.markdown(f"- `{fragment}` - {why}")

    st.markdown('<div class="eyebrow">3. Know before you run</div>', unsafe_allow_html=True)
    st.write(s["watch_out"])
    if s["queues"]:
        st.write("**Queues**")
        for q, note in s["queues"]:
            st.markdown(f"- `{q}` - {note}")
    if s["jupyter"]:
        st.write(f"**Notebooks:** {s['jupyter_note']}")
    elif s["jupyter_note"]:
        st.caption(s["jupyter_note"])

# --- Full specs ------------------------------------------------------------
with tab_specs:
    s = top["sys"]
    left, right = st.columns(2)
    with left:
        st.write(f"**Vendor / platform:** {s['vendor']}")
        st.write(f"**Nodes:** {s['nodes'] if s['nodes'] else 'not published'}")
        st.write(f"**CPU:** {s['cpu']}")
        if s["cores_per_node"]:
            st.write(f"**Cores per node:** {s['cores_per_node']} physical "
                     f"({s['threads_per_node']} threads)")
        if s["cpu_mem_gb"]:
            st.write(f"**CPU memory per node:** {s['cpu_mem_gb']} GB - {s['cpu_mem_note']}")
    with right:
        st.write(f"**Accelerator:** {s['accel_model']}")
        if s["accel_note"]:
            st.write(f"**Note:** {s['accel_note']}")
        st.write(f"**Interconnect:** {s['interconnect']}")
        st.write(f"**Node-local storage:** {s['node_storage']}")
        st.write(f"**Shared file systems:** {s['facility_storage']}")
        st.write(f"**Allocation:** {s['allocation']}")

    if s["software"]:
        st.write("**Software environment**")
        st.code(s["software"], language="bash")

    st.write("**Documentation**")
    for label, url in s["links"].items():
        st.markdown(f"- [{label}]({url})")
    st.caption("Verified " + LAST_VERIFIED + " against: " + ", ".join(s["sources"]))

# --- Compare ---------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="eyebrow">Side-by-side comparison</div>', unsafe_allow_html=True)
    all_names = [s["name"] for s in SYSTEMS]
    default_names = [r["sys"]["name"] for r in (clean if clean else results)[:2]]
    picks = st.multiselect(
        "Pick two or three systems",
        all_names,
        default=default_names,
        max_selections=3,
        help="Defaults to the top two systems for your current requirements.",
    )

    if len(picks) < 2:
        st.info("Pick at least two systems to compare.")
    else:
        chosen = [s for s in SYSTEMS if s["name"] in picks]

        def fmt(v):
            return v if v not in (None, "", 0) else "-"

        COMPARE_FIELDS = [
            ("Match score (your settings)",
             lambda s: f"{by_name[s['name']]['score']} pts"
                       + ("" if not by_name[s['name']]["misses"]
                          else f"  ({len(by_name[s['name']]['misses'])} requirement(s) missed)")),
            ("Family", lambda s: s["family"]),
            ("Status", lambda s: s["status"]),
            ("Nodes", lambda s: f"{s['nodes']:,}" if s["nodes"] else "not published"),
            ("Accelerator", lambda s: s["accel_model"]),
            ("Accelerators / node", lambda s: fmt(s["accel_per_node"])),
            ("Accelerators total", lambda s: f"{accel_total(s):,}" if accel_total(s) else "-"),
            ("Accel memory / node (GB)", lambda s: fmt(s["accel_mem_gb"])),
            ("CPU", lambda s: s["cpu"]),
            ("Cores / node (physical)", lambda s: fmt(s["cores_per_node"])),
            ("CPU memory / node (GB)", lambda s: fmt(s["cpu_mem_gb"])),
            ("Interconnect", lambda s: s["interconnect"]),
            ("Node-local storage", lambda s: s["node_storage"]),
            ("Shared file systems", lambda s: s["facility_storage"]),
            ("Scheduler", lambda s: s["scheduler"]),
            ("Login host", lambda s: s["login"] or "-"),
            ("JupyterHub", lambda s: "yes" if s["jupyter"] else "no"),
            ("Allocation", lambda s: s["allocation"]),
            ("Watch out", lambda s: s["watch_out"]),
        ]

        compare_df = pd.DataFrame(
            {s["name"]: [str(fn(s)) for _, fn in COMPARE_FIELDS] for s in chosen},
            index=[label for label, _ in COMPARE_FIELDS],
        )
        st.dataframe(compare_df, width="stretch", height=680)

        cols = st.columns(len(chosen))
        for col, s in zip(cols, chosen):
            with col:
                st.markdown(f"**{s['name']}** - {s['tagline']}")
                with st.expander("Score breakdown"):
                    render_breakdown(by_name[s["name"]])

# --- All systems -----------------------------------------------------------
with tab_all:
    st.markdown('<div class="eyebrow">Ranked for your requirements</div>', unsafe_allow_html=True)

    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        query = st.text_input(
            "Search", placeholder="e.g. A100, SLURM, wafer, HPE...",
            label_visibility="collapsed",
        )
    with fcol2:
        fam_filter = st.multiselect(
            "Family", sorted({s["family"] for s in SYSTEMS}), placeholder="Family",
            label_visibility="collapsed",
        )
    with fcol3:
        status_filter = st.multiselect(
            "Status", sorted({s["status"] for s in SYSTEMS}), placeholder="Status",
            label_visibility="collapsed",
        )

    shown = clean if (hide_misses and clean) else results
    if query:
        shown = [r for r in shown if matches_query(r["sys"], query)]
    if fam_filter:
        shown = [r for r in shown if r["sys"]["family"] in fam_filter]
    if status_filter:
        shown = [r for r in shown if r["sys"]["status"] in status_filter]

    if not shown:
        st.info("No systems match the current search and filters.")

    for r in shown:
        s = r["sys"]
        flag = "" if not r["misses"] else "  -  misses: " + ", ".join(r["misses"])
        with st.expander(f"{s['name']}  ({s['family']})  -  {r['score']} pts{flag}"):
            st.write(s["tagline"])
            render_specstrip(s)
            st.write(s["watch_out"])
            render_breakdown(r)
            st.markdown(" &middot; ".join(
                f"[{label}]({url})" for label, url in s["links"].items()
            ))

    st.markdown('<div class="eyebrow">Side by side</div>', unsafe_allow_html=True)
    spec_df = pd.DataFrame([spec_row(s) for s in SYSTEMS])
    for col in ("Nodes", "Accelerators/node", "Cores/node",
                "CPU mem/node (GB)", "Accel mem/node (GB)"):
        spec_df[col] = spec_df[col].astype("Int64")   # nullable ints, Arrow-clean
    st.dataframe(spec_df, width="stretch", hide_index=True)
    st.download_button(
        "Download the spec table (CSV)",
        spec_df.to_csv(index=False),
        file_name="alcf_systems.csv",
        mime="text/csv",
    )
    st.caption(
        "Node-local scratch and shared file systems are listed per system rather than in "
        "this table - they are different resources and are not comparable as one number."
    )

# --- New here? ---------------------------------------------------------------
with tab_new:
    gcol, ccol = st.columns([3, 2], gap="large")

    with gcol:
        st.markdown('<div class="eyebrow">Glossary - HPC words, plainly</div>',
                    unsafe_allow_html=True)
        gquery = st.text_input(
            "Filter terms", placeholder="Filter terms...",
            label_visibility="collapsed", key="glossary_query",
        )
        terms = {
            t: d for t, d in GLOSSARY.items()
            if not gquery or gquery.lower() in t.lower() or gquery.lower() in d.lower()
        }
        if not terms:
            st.info("No glossary term matches that filter.")
        for term, definition in terms.items():
            with st.expander(term):
                st.write(definition)

    with ccol:
        st.markdown('<div class="eyebrow">Your first week - checklist</div>',
                    unsafe_allow_html=True)
        st.caption("Ticks live in this browser session. Work top to bottom.")
        done = 0
        for i, (title, detail) in enumerate(CHECKLIST):
            checked = st.checkbox(title, key=f"chk_{i}")
            if checked:
                done += 1
            st.caption(detail)
        st.progress(done / len(CHECKLIST))
        if done == len(CHECKLIST):
            st.success("All set. Submit something real - and keep the debug queue "
                       "in your pocket for anything new.")
        else:
            st.markdown(f'<div class="scoreline">{done} of {len(CHECKLIST)} done</div>',
                        unsafe_allow_html=True)

# --- Access ----------------------------------------------------------------
with tab_access:
    st.markdown('<div class="eyebrow">If you have no account yet</div>', unsafe_allow_html=True)
    st.markdown(
        """
1. **Get a project.** Director's Discretionary is the usual starting point and is open
   year-round, including for the AI Testbed. INCITE and ALCC run on annual calls and are
   for large, established campaigns.
2. **Request an ALCF account** once the project is approved, and join your PI's project.
3. **Set up your token.** Every login needs a CRYPTOCard or MobilePASS+ passcode.
4. **Move your data** with Globus. The AI Testbed uses its own endpoints
   (`alcf#ai_testbed_home`, `alcf#ai_testbed_projects`).
5. **Run something small first** - the debug queues exist for exactly this.

**Two things that need no allocation at all:** the Inference Service (including Metis) is
open to every ALCF user, and the AskALCF chatbot will answer questions about queues,
modules, and job scripts before you commit to a system.
        """
    )
    for label, url in ACCESS_LINKS.items():
        st.markdown(f"- [{label}]({url})")

st.divider()
st.markdown(
    f'<div class="caveat">Specifications verified {LAST_VERIFIED} against docs.alcf.anl.gov and '
    'alcf.anl.gov. Hardware, queues, and policies change - confirm with the linked documentation '
    'or support@alcf.anl.gov before making capacity or proposal decisions. '
    'Systems marked "not in production yet" are announced only; per-node details for Minerva and '
    'Janus have not been published.</div>',
    unsafe_allow_html=True,
)


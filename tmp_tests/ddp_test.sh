#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --account=geminigrp
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lion397/codes/image-to-l-system/tmp_tests/ddp_test_%j.log
cd /home/lion397/codes/image-to-l-system
export PYTHONPATH=.
/home/lion397/.conda/envs/digital-crops/bin/torchrun --nproc_per_node=2 --master_port=29602 tmp_tests/ddp_smoke.py

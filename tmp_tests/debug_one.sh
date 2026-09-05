#!/bin/bash
#SBATCH --job-name=dbg1
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --account=geminigrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/lion397/codes/image-to-l-system/tmp_tests/debug_one_%j.log
cd /home/lion397/codes/image-to-l-system
export PYTHONPATH=.
/home/lion397/.conda/envs/digital-crops/bin/python tmp_tests/debug_one.py

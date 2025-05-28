#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=0
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=1
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=2
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=3
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=4
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=5
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=6
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=7
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=8
CUDA_VISIBLE_DEVICES=0 python loop_save.py --workdir=./ --config=configs/EI_leaky.py --seed=9
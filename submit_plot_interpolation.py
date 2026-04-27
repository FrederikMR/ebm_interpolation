#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < plot_interpolant.sh")
    
    return

#%% Generate jobs

def generate_job(model_name, metric_type):

    with open ('plot_interpolant.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J interpolant_{model_name}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=32GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o ../output_folder/output_%J.out 
    #BSUB -e ../error_folder/error_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    source /work3/fmry/miniconda3/bin/activate
    conda activate section_talk_03_03_26
    
    python plot_interpolation.py \
        --dataset 'afhq' \
        --data_root '/work3/fmry/Data/afhq/stargan-v1/data/' \
        --ebm_root 'ebm_your_saving_path' \
        --figure_path '/work3/fmry/projects/RiemannEBM/figures/' \
        --save_root '/work3/fmry/projects/RiemannEBM/RiemannEBM/saved_interpolants/' \
        --ebm_ckpt '/work3/fmry/projects/RiemannEBM/RiemannEBM/saved_models/afhq/model_afhq/last.model' \
        --batch_size 12 \
        --t_steps 50 \
        --lr 1e-4 \
        --ebm_multiplier 1 \
        --nb_iteration 200000 \
        --model_name {model_name} \
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):

    metric_types = ['conf_ebm_invp',
                    'conf_ebm_logp',
                    'diag_ebm_logp',
                    'diag_ebm_invp',
                    ## 'diag_rbf_invp',
                    #'conf_rbf_invp',
                    'diag_land_invp',
                    'full_ebm_invp',
                    'full_ebm_logp',
                    #'newton_riemann_logp',
                    #'newton_riemann_invp',
                    'newton_riemann_mlogp',
                    'newton_riemann_minvp',
                    ]

    run_model(metric_types, wait_time)

    return
                            
def run_model(metric_types, wait_time):

    time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
    generate_job(metric_type = metric_types,
                model_name = 'plotting', 
                )
    try:
        submit_job()
    except:
        time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
        try:
            submit_job()
        except:
            print(f"Job script failed!")


#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)

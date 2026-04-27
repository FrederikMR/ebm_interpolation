    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J interpolant_plotting
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
    
    python plot_interpolation.py         --dataset 'afhq'         --data_root '/work3/fmry/Data/afhq/stargan-v1/data/'         --ebm_root 'ebm_your_saving_path'         --figure_path '/work3/fmry/projects/RiemannEBM/figures/'         --save_root '/work3/fmry/projects/RiemannEBM/RiemannEBM/saved_interpolants/'         --ebm_ckpt '/work3/fmry/projects/RiemannEBM/RiemannEBM/saved_models/afhq/model_afhq/last.model'         --batch_size 12         --t_steps 50         --lr 1e-4         --ebm_multiplier 1         --nb_iteration 200000         --model_name plotting     
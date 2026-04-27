    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J train_ebm
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=16GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o ../output_folder/output_%J.out 
    #BSUB -e ../error_folder/error_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    source /work3/fmry/miniconda3/bin/activate
    conda activate section_talk_03_03_26
    
    python3 train_latent_ebm.py \
        --data_root '/work3/fmry/Data/afhq/stargan-v1/data/' \
        --save_root '/work3/fmry/projects/RiemannEBM/RiemannEBM/saved_models/' \
        --init_type 'normal_01' \
        --training 'cdsm' \
        --epoch 3000 \
        --n_steps 50 \
        --buffer_size 10000 \
        --energy_func 'VanillaNet_ELU_2' \
        --model_name 'model_afhq' \
    
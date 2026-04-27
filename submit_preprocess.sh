    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J preprocess_ebm
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
    
    python3 dataset.py \
        --dataset afhq \
    
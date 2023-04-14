set -xe
PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../..
source deactivate
conda activate openfold

wandb offline

train_strategy=horovod

# ## VH-VL 
train_data_dir=   # specify the dir contains *.pdb
train_alignment_dir=  # a dir to save template and features.pkl of training sequence


train_data_csv_path=
val_data_csv_path=


output_dir=${PROJECT_DIR}/ablation_blocks/block32 # a dir to save states and models
mkdir -p ${output_dir}

train_epoch_len=1000 # virtual length of each training epoch, which affects frequency of validation & checkpointing

config_preset=multimer 
config_yaml_fpth=${PROJECT_DIR}/openfold/yaml_lib/train/xTrimoDock_stage1.yaml
gradient_clip_val_for_horovod=0.1


resume_from_ckpt=
resume_model_weights_only=False
cuda_device=0
CUDA_VISIBLE_DEVICES=${cuda_device} python ${PROJECT_DIR}/train_openfold.py \
    --train_data_dir=${train_data_dir} \
    --train_alignment_dir=${train_alignment_dir} \
    --train_data_csv_path=${train_data_csv_path} \
    --val_data_csv_path=${val_data_csv_path} \
    --config_preset=${config_preset} \
    --train_strategy=${train_strategy} \
    --output_dir=${output_dir} \
    --train_epoch_len=${train_epoch_len} \
    --checkpoint_every_epoch \
    --config_yaml_fpth=${config_yaml_fpth} \
    --gradient_clip_val_for_horovod=${gradient_clip_val_for_horovod} \
    --wandb \
    --resume_from_ckpt=${resume_from_ckpt} \
    --resume_model_weights_only=${resume_model_weights_only} \
    --prof_memory_main\
    --flow_attn \
    --pair_factor \
    


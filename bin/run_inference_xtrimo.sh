#!/bin/bash

set -xe
PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../..
source deactivate
conda activate openfold

cuda_device=0  # check available cuda device before running this bash

fasta_paths=/nfs_baoding_ai/sunning_2022/DB5_fasta
tmpl_pdb_map=None

msa_method_config=mmseqs_jackhmmer_uniprot_886cff956097ef901040c3a6bbc68942  # AF2_0df8042a5fe9a829368522d695328450, multimer_f89063cbe31815418d57ceaba36caeb0, mmseqs_jackhmmer_uniprot_886cff956097ef901040c3a6bbc68942
model_preset=multimer  # monomer, multimer
config_yaml_fpth=${PROJECT_DIR}/openfold/yaml_lib/train/xTrimoDock_stage1.yaml


param_dir=/
param_type=torch  # only support params trained on k8s(horovod)
num_model=1  # integer, [1, 5]
max_template_date=2030-07-10
saved_emds=none  # none;msa_first_row,single,structure_module,msa,pair
specific_tmpl_mode=max  # max, all
max_subsequence_ratio_p=0.95  # Exclude any exact matches with this much overlap. min_align_ratio: Minimum overlap between the template and query.
min_align_ratio_p=0.1  # Minimum overlap between the template and query.

output_dir=${PROJECT_DIR}/DB_results/baseline # specify output directory

mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=${cuda_device} python ${PROJECT_DIR}/run_pretrained_openfold.py \
    --max_template_date=${max_template_date}  \
    --msa_method_config=${msa_method_config} \
    --model_preset=${model_preset} \
    --fasta_paths=${fasta_paths} \
    --tmpl_pdb_map=${tmpl_pdb_map} \
    --output_dir=${output_dir} \
    --num_model=${num_model} \
    --saved_emds=${saved_emds} \
    --specific_tmpl_mode=${specific_tmpl_mode} \
    --max_subsequence_ratio_p=${max_subsequence_ratio_p} \
    --min_align_ratio_p=${min_align_ratio_p} \
    --param_type=${param_type} \
    --param_dir=${param_dir} \
    --config_yaml_fpth=${config_yaml_fpth} \
    --cuda \
    --cuda_for_amber \
    --skip_amber_relax \
    --reproduce \
    --dev \
    --pair_factor \
    --flow_attn \


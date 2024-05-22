


# Liteformer

A PyTorch reproduction of Liteformer based on DeepMind's
[AlphaFold 2](https://github.com/deepmind/alphafold).


## Installation (Linux)

All Python dependencies are specified in `environment.yml`. 

For convenience, we provide a script that installs Miniconda locally, creates a 
`conda` virtual environment, installs all Python dependencies, and downloads
useful resources, including both sets of model parameters. Run:

```bash
scripts/install_third_party_dependencies.sh
```

To activate the environment, run:

```bash
source scripts/activate_conda_env.sh
```

To deactivate it, run:

```bash
source scripts/deactivate_conda_env.sh
```

With the environment active, compile OpenFold's CUDA kernels with

```bash
python3 setup.py install
```

## Data Preparation

CASP14 dataset can be downloaded from [casp14](https://predictioncenter.org/download_area/CASP14/).  

DB5.5 dataset can be downloaded from [db55](https://github.com/octavian-ganea/equidock_public).  

BCR, VH-VL, AB-AG datasets can be downloaded from [pdb](https://www.rcsb.org/#Category-download).

### Pretrained Models


## Train

The sample training script is /bin/run_train.sh . "flow_attn" denotes using BAA and "pair_factor" denotes using VLD. 

If you intend to train Lightformer, you can run:

```bash
python ${PROJECT_DIR}/train_openfold.py \
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
    --resume_from_ckpt=${resume_from_ckpt} \
    --resume_model_weights_only=${resume_model_weights_only} \
    --flow_attn \
    --pair_factor \
```

If you intend to train Alphafold2, you can run:

For the latter, run:

```bash
python ${PROJECT_DIR}/train_openfold.py \
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
    --resume_from_ckpt=${resume_from_ckpt} \
    --resume_model_weights_only=${resume_model_weights_only} \
```


## Inference

To run inference on a sequence or multiple sequences using a set of Lightformer's
pretrained parameters, run e.g.:

```bash
python ${PROJECT_DIR}/run_pretrained_openfold.py \
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
```



```

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Contributing

If you encounter problems using OpenFold, feel free to create an issue! We also
welcome pull requests from the community.

## Citing this work

Please cite our paper:

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {AlphaFold2 revolutionized structural biology with the ability to predict protein structures with exceptionally high accuracy. Its implementation, however, lacks the code and data required to train new models. These are necessary to (i) tackle new tasks, like protein-ligand complex structure prediction, (ii) investigate the process by which the model learns, which remains poorly understood, and (iii) assess the model{\textquoteright}s generalization capacity to unseen regions of fold space. Here we report OpenFold, a fast, memory-efficient, and trainable implementation of AlphaFold2, and OpenProteinSet, the largest public database of protein multiple sequence alignments. We use OpenProteinSet to train OpenFold from scratch, fully matching the accuracy of AlphaFold2. Having established parity, we assess OpenFold{\textquoteright}s capacity to generalize across fold space by retraining it using carefully designed datasets. We find that OpenFold is remarkably robust at generalizing despite extreme reductions in training set size and diversity, including near-complete elisions of classes of secondary structure elements. By analyzing intermediate structures produced by OpenFold during training, we also gain surprising insights into the manner in which the model learns to fold proteins, discovering that spatial dimensions are learned sequentially. Taken together, our studies demonstrate the power and utility of OpenFold, which we believe will prove to be a crucial new resource for the protein modeling community.},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

Any work that cites OpenFold should also cite AlphaFold.

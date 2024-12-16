@echo off

:: Set the working directory
:: cd <project PYTHONPATH>
cd C:\Users\10594\PycharmProjects\ivt\IVT

:: Activate the conda environment
::call conda activate <your environment>
call .venv\Scripts\activate


:: Set necessary environment variables
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0

::python runnables/train_iv_multi.py -m +dataset=iv_cancer_sim +backbone=ivt +backbone/ivt_hparams/cancer_sim_domain_conf='1' exp.seed=10 > abc.log
python runnables/train_iv_multi.py -m +dataset=iv_cancer_sim +backbone=ivt +backbone/ivt_hparams/cancer_sim_domain_conf='1' exp.seed=10 > abc.log
::python runnables/train_iv_multi.py -m +dataset=cancer_sim +backbone=ivct +backbone/ivct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=40 dataset.radio_iv_influence_ratio=40
::python runnables/train_iv_multi.py -m +dataset=cancer_sim +backbone=ivct +backbone/ivct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=10 dataset.radio_iv_influence_ratio=10
::python runnables/train_iv_multi.py -m +dataset=cancer_sim +backbone=ivct +backbone/ivct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=5 dataset.radio_iv_influence_ratio=5
::python runnables/train_iv_multi.py -m +dataset=cancer_sim +backbone=ivct +backbone/ivct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=1 dataset.radio_iv_influence_ratio=1

::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10
::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=80 dataset.radio_iv_influence_ratio=80
::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=40 dataset.radio_iv_influence_ratio=40
::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=10 dataset.radio_iv_influence_ratio=10
::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10 dataset.chemo_iv_influence_ratio=1 dataset.radio_iv_influence_ratio=1
::python runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf='1' exp.seed=10
::python runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn +backbone/crn_hparams/cancer_sim_domain_conf='1' exp.seed=10

::python runnables/train_rmsn.py -m +dataset=cancer_sim +backbone=rmsn +backbone/rmsn_hparams/cancer_sim_domain_conf='1' exp.seed=10
::python runnables/train_gnet.py -m +dataset=climate_real +backbone=gnet +backbone/gnet_hparams/climate_real=prate exp.seed=10
echo All runs complete
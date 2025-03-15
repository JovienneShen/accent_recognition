# Introduction
This repository implements a two-class accent classifier (UK vs. US English), adapted from [Juan P. Zuluaga’s accent-recog-slt2022](https://github.com/JuanPZuluaga/accent-recog-slt2022).

Our model achieves 82 % accuracy distinguishing “England” and “US” accents.

In the sections below you’ll find step-by-step instructions—starting from dataset preparation through to training, evaluation, and inference—so you can reproduce every stage of our workflow.

# Installation of environment

1. Install [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network)

The project need CUDA to support the training and inference. Download CUDA 11.7 from the link above and choose default configuration during the installation.

After the installation, run 'nvcc --version' in terminal to check the installation of CUDA.

2. Install [Miniconda3 Windows 64-bit](https://docs.anaconda.com/miniconda/)

Download from the link and install the miniconda with the default configuration.

3. Clone this project

Clone the project with following command.
```
git clone https://github.com/JovienneShen/accent_recognition.git
```

4. Create python environment

Open Anaconda Prompt in the project directory, then enter following command to create a Python environment:

```
conda create -n accent_recog python==3.10
conda activate accent_recog
pip install -r requirements.txt
pip install -e .
```

If you got problem with pytorch version, please reinstall Pytorch and speechbrain with following version.
```
pip install speechbrain==0.5.13 matplotlib
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
This environment is only for training and testing models with our dataset.

Then we need to create another environment, first let's deactivate the previous environment.
```
deactivate
```
Then run following commands to create a new environment:
```
conda create -n accent_recog_inf python==3.10
conda activate accent_recog_inf
pip install -r model_inference/requirements_inf.txt
```
This environment is for reloading trained models and inference it on given data and audios.

# Data preparation

The data used in this prject is [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)

## Download dataset

To download the dataset we need to run the  following command to download the dataset.
In **accent_recog** environment
```
cd speechbrain\accent-recog-slt2022\CommonAccent
python download_data_hf.py --language "en" data/cv_11/
```
It takes about one day to finish downloading.

Or you can run download_cv11.py in the root directory of the project.
It also takes about one day ...
```
python download_cv11.py
```

Then copy all folders named as "en_xx_xx" from the folder 
`C:\Users\<user_name>\.cache\huggingface\datasets\downloads\extracted`
to `accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\data\cv_11_raw`

You can skip the following steps as we already provided csv files in `accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\accent_id\data`.

If you want to know how we generate them, please follows the next steps.

## Preprocess dataset

To generate the train/dev/test data list csv files, we need to run following command in `speechbrain\accent-recog-slt2022\CommonAccent` to preprocess the dataset:
In **accent_recog** environment
```
python common_accent_prepare.py --language "en" data/cv_11 data/
```
Then copy generated .csv files from
`accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\data`
into
`accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\accent_id\data`

## Filter dataset

We used a filtering strategy to refined the dataset, here is the guide to execute the filtering process.

In **accent_recog_inf** environment, run following commands:
```
cd model_inference
python filter_data_based_on_model.py
```
Then you need to replace the training csv file with the generated csv file.

# Training and inference with dataset
Here is the [reference](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa) of the training and inference of the project. Here we provided our training and inference steps.

## Train

In `accent_recog/speechbrain/accent-recog-slt2022/CommonAccent/accent_id` run following command to train our English vs US accent model.
In **accent_recog** environment:
```
python train.py hparams/train_ecapa_tdnn_2accents.yaml
```

## Inference

Before this make sure that you copies dev.csv test.csv from `accent_recog/speechbrain/accent-recog-slt2022/CommonAccent/data` to `accent_recog/speechbrain/accent-recog-slt2022/CommonAccent/accent_id/data`.

Then create a dummy.csv from test.csv in accent-recog-slt2022/CommonAccent/accent_id/data, just keep one row is enough.

In the same directory, run following command to inference the model on test set.
In **accent_recog** environment:
```
python inference.py hparams/inference_ecapa_tdnn_en_us_70k.yaml
```

# Pretrained models

The pretrained models are saved in `speechbrain\accent-recog-slt2022\CommonAccent\accent_id\results`,
named as follow:

- British vs US 70k dataset
    Model : `speechbrain/accent-recog-slt2022/CommonAccent/accent_id/results/ECAPA-TDNN-241229-70k`
    Evaluation result : `speechbrain/accent-recog-slt2022/CommonAccent/accent_id/results/analysis_ECAPA-TDNN-241229_70k`
- British vs US 70k dataset with filtration
    Model : `speechbrain/accent-recog-slt2022/CommonAccent/accent_id/results/ECAPA-TDNN-241229-70k-filtered-0.7`
    Evaluation result : `speechbrain/accent-recog-slt2022/CommonAccent/accent_id/results/analysis_ECAPA-TDNN-241229_70k_filtered-0.7`

# Inference on audio files

Save all audio files into `model_inference\audios` directory, then run following commands to inference a model on them and get the results.
In **accent_recog_inf** environment:
```
python inference_file.py
```
The result will be saved in `model_inference\outputs\accent_detection_results.csv`

For changing details about the inference, please go deep into the codes of the file : `model_inference/inference_file.py`
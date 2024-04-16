# SSTC2024_baseline_system

## 0. Introduction

This repository is the SSTC2022 baseline system, including:

* Environment preparation
* Data preparation
* Model training
* Embedding extracting
* Performance calculating

Please visit https://sstc-challenge.github.io/ for more information about the challenge.

## 1. System Pipeline

#### Step 1. Environment preparation

##### Conda

We recommend installing dependencies in the conda environment

```
conda create -y -n baseline python=3.8
conda activate baseline
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install nemo_toolkit['all']==1.14.0
```

#### Step 2. Data preparation

The system adopts the online data augmentation method for model training. Please prepare the <a href="https://www.openslr.org/17/">MUSAN</a> and <a href="https://www.openslr.org/17/">RIR_NOISES</a>  dataset and modify the path of './data/musan/' and './data/rir_noise/' files as your data path. 

For voice conversion, we utilize <a href="http://www.openslr.org/12">Librispeech</a> as the source speaker dataset and <a href="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/">voxceleb1&2</a> as the target speaker dataset. The converted speech datasets can be download in https://sstc-challenge.github.io/.

The data preparation file follows the Kaldi form that participants need "wav.scp", "utt2spk", "spk2utt" files for training dir, and "wav.scp" and "trials" for valuation dir. The "./data/vox2dev" shows the training example files and "./data/vox1-O" shows the valuation example files. There are five data dir need to be prepared in the baseline system recipe:

```
# target speaker dataset
./data/Vox2dev/
    ./wav.scp
    ./utt2spk
    ./spk2utt

# source speaker dataset
./data/librispeech/train/
    ./wav.scp
    ./utt2spk
    ./spk2utt

# training dataset
./data/vcdata/
    ./train_1/
        ./wav.scp
        ./utt2spk
    ./train_2/
        ./wav.scp
        ./utt2spk
    ...
    ./train_8/
        ./wav.scp
        ./utt2spk

# development dataset
./data/vc-dev/
    ./dev_1.scp
    ./dev_2.scp
    ...
    ./dev_12.scp
    ./dev_trials  # with keys

# evaluation dataset
./data/vc-eval/
    ./eval_1.scp
    ./eval_2.scp
    ...
    ./eval_k.scp
    ./eval_trials  # without keys
```

#### Step 3. Model training

We employ a pre-training strategy on the VoxCeleb2 development set to allow the model to initially learn general features, so as to obtain better performance on subsequent tasks.

The following are the pre-trained model (half small MFA_Conformer) results on vox1-O.

| Vox1-O (EER) | Download Link         |
| ------------ | --------------------- |
| 1.145%       | [Download](https://drive.google.com/file/d/1TO0NuPJJkXR6TfUupMP9XQyUBdSdGCva/view?usp=drive_link) |

Running:

```
python train.py --save_dir 8vc \
    --data_name train_1 train_2 train_3 train_4 train_5 train_6 train_7 train_8 \
    --warmup_epochs 1 --dur_range 2 2 \
    --val_data_name vc-dev \
    --batch_size 512 --workers 40 \
    --mels 80 --fft 512 \
    --model ConformerMFA --embd_dim 256 \
    --classifier ArcFace --angular_m 0.2 --angular_s 32 --dropout 0 \
    --gpu 0,1,2,3 --epochs 25  --start_epoch 0 --lr 0.001 &
```

#### Step 4. Valuation model

There are three modes for scoring.py,

```
# Extract speaker embedding and compute the EER and mDCF 
scoring = True
onlyscoring = False

# Extract speaker embedding
scoring = False
onlyscoring = False

# Compute EER and mDCF
scoring = False/True
onlyscoring = True
```

Running:

```
python scoring.py --save_dir 8vc \
    --val_data_name vc-dev --vc_method dev_1 --val_save_name dev_1  --model_num 24 \
    --onlyscore False --scoring True --trials dev_trials \
    --gpu 0 &
```

#### Performance of baseline system

| Dev-1     | Dev-2     | Dev-3     | Dev-4      | Dev-5      | Dev-6      |
| --------- | --------- | --------- | ---------- | ---------- | ---------- |
| 9.397%    | 8.619%    | 7.671%    | 7.594%     | 7.507%     | 12.885%    |
| **Dev-7** | **Dev-8** | **Dev-9** | **Dev-10** | **Dev-11** | **Dev-12** |
| 32.484%   | 28.795%    | 34.045%   | 45.772%    | 17.209%    | 20.808%    |

**Download Link:  [Download](https://drive.google.com/file/d/1K5S7aUVUTqgPVgqQx7fpQWO1ZbYuJ6MX/view?usp=drive_link).**



##### **One tip: **

**you may use the method category information to help improve the performance of your model.**

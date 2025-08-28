# ASGPF: A Self-Supervised Pre-Training Method for Seizure Classification

Yue Hu, Jian Liu, Wenli Zhang, Yi Sui, Qingyue Meng and Rencheng Sun.

---

## Background
Objective: Epileptic seizure classification using EEG signals remains a significant challenge due to complex spatial-temporal dependencies, limited labeled data, and severe class imbalance. Methods: We propose a self-supervised learning framework, ASGPF (Adaptive Spatio-Graph Pretraining Framework), for EEG-based seizure classification. At its core is a novel Spatio-Graph Learning Cell (SGLC), which integrates a Graph Learning module to dynamically construct EEG topology, a Gated Graph Neural Network to extract spatial features across EEG channels, and a Gated Recurrent Unit to capture long-term temporal dependencies. ASGPF uses self-supervised sequence-to-sequence pretraining on unlabeled EEG to learn robust representations, enabling accurate seizure classification with a lightweight model that consists of the pretrained encoder and a simple prediction layer. Results: Extensive experiments on the TUSZ dataset demonstrate that our method significantly outperforms current state-of-the-art approaches, achieving weighted F1-scores of 83.8% for four-class and 73.5% for eight-class seizure classification tasks, respectively. Notably, with only 25% of labeled data, the proposed model achieves comparable performance to the best baseline trained on 75% of data, validating the effectiveness of ASGPF under data scarcity and class imbalance. Conclusion: ASGPF effectively learns discriminative EEG representations through adaptive spatial-temporal modeling and self-supervised pretraining, enabling accurate seizure classification with minimal labeled data. Significance: This work introduces a data-efficient EEG analysis framework for seizure classification, enabling accurate prediction with minimal labeled data and showing strong potential for clinical application in resource- and label-constrained environments.

---
## Data

We use the Temple University Seizure Corpus (TUSZ) v1.5.2 in this study. 

In this study, we exclude five patients from the test set who exist in both the official TUSZ train and test sets. You can find the list of excluded patients' IDs in `./data_tusz/excluded_test_patients.txt`.

---

## Conda Environment Setup

On terminal, run the following:
```
conda env create -f eeg_asgpf.yml
conda activate eeg_asgpf
```

---

## Preprocessing
The preprocessing step resamples all EEG signals to 200Hz, and saves the resampled signals in 19 EEG channels as `h5` files.

On terminal, run the following:
```
python ./data/resample_signals.py --raw_edf_dir <tusz-data-dir> --save_dir <resampled-dir>
```
where `<tusz-data-dir>` is the directory where the downloaded TUSZ v1.5.2 data are located, and `<resampled-dir>` is the directory where the resampled signals will be saved.

#### Optional Preprocessing
Note that the remaining preprocessing step in this study --- Fourier transform on short sliding windows, is handled by dataloaders. You can (optionally) perform this preprocessing step prior to model training to accelerate the training.

Preprocessing for seizure classification:
```
python ./data/preprocess_classification.py --resampled_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --output_dir <preproc-dir> --clip_len <clip-len> --time_step_size 1 --is_fft
```

---
## Experiments

### Seizure Type Classification
To train seizure type classification from scratch using **correlation-based EEG graph**, run: 
```
python train.py --input_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --save_dir <save-dir> --graph_type individual --max_seq_len <clip-len> --do_train --num_epochs 60 --task classification --model_name sglc --metric_name F1 --use_fft --lr_init 3e-4 --num_rnn_layers 2 --rnn_units 64 --max_diffusion_step 2 --num_classes 4 --data_augment --dropout 0.5 
```


### Self-Supervised Pre-Training
To train self-supervised next time period prediction using **correlation-based EEG graph**, run: 
```
python train_ssl.py --input_dir <resampled-dir> --raw_data_dir <tusz-data-dir> --save_dir <save-dir> --graph_type individual --max_seq_len <clip-len> --output_seq_len 12 --do_train --num_epochs 350 --task 'SS-pre-training' --ssl_model asgpf --metric_name loss --use_fft --lr_init 5e-4 --num_rnn_layers 3 --rnn_units 64 --max_diffusion_step 2 --data_augment
```

### Fine-Tuning for Seizure Detection & Seizure Type Classification
To fine-tune seizure type classification models from self-supervised pre-training, **add** the following additional arguments:
```
--fine_tune --load_model_path <pretrained-model-checkpoint>
```
---

## Statement
The code repository will be made publicly available upon formal acceptance of the manuscript through a trusted archival platform.

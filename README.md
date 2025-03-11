# ASGPF: A Self-Supervised Pre-Training Method for Seizure Classification

Yue Hu, Jian Liu, Yi Sui, Wenli Zhang, Rencheng Sun, Qingyue Meng.

---

## Background
To address the challenges of inter-subject heterogeneity, data scarcity, and class imbalance in epileptic seizure classification, we propose a self-supervised pretraining framework with three key innovations:
(1)We propose a Spatio-Graph Learning Cell (SGLC), which consists of a Graph Learning (GL) module, a Gated Graph Neural Network (GGNN) and a Gated Recurrent Unit (GRU). The GL module iteratively optimizes the graph structure, the GGNN captures intricate spatial dependencies between EEG channels, and the GRU models dynamic temporal variations and long-term dependencies in EEG signals. This design enables the SGLC not only to address inter-subject heterogeneity in EEG signals but also to effectively capture complex spatiotemporal correlations among EEG channel signals;
(2)We design an Adaptive Spatio-Graph Pretraining Framework (ASGPF), which constructs an encoder-decoder architecture using the SGLC. The framework utilizes unlabeled EEG data for sequence-to-sequence pretraining and builds a multi-class seizure classification model based on the encoder combined with a prediction layer. The pretraining enables the encoder to efficiently learn latent features from vast amounts of unlabeled EEG data, ensuring that the classification model maintains excellent performance even with limited labeled samples and class imbalance;
(3)Experiments on the TUSZ dataset for four-class and eight-class seizure classification demonstrate that the encoder combined with the prediction layer achieves weighted F1-scores of 76.3% and 63.5%, respectively. When integrated with pretraining, performance improves to 83.8% (four-class) and 73.5% (eight-class), significantly surpassing baseline methods and demonstrating that ASGPF effectively mitigates class imbalance. Notably, with just 25% of the training data, ASGPF attains classification performance comparable to baseline models trained on 75% of the data, validating its adaptability to data scarcity.

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

## Introduction

This project aims at completing the Kaggle BCI Challenge @ NER 2015 : https://www.kaggle.com/c/inria-bci-challenge/overview

It consists in discriminating wrong from good feedbacks from a P300 Speller experiment. 
I used two different approaches in this project and compare them :
1) Extracting relevant and then fit machine learning models
2) Training deep neural networks (EEGNet and ERPENet) on raw data

## Feature extraction

### Reproduce features extraction (may be long) :

```shell script
$ python3 util/features_extractions.py --name_features feature_name --ica 0 --spectral 0 --normalized 0 --setup "eegnet"

    options:
          --name_features (str) : name of the features, please refer to the list below (required)
          --ica (int) : Perform ICA on EEG data or not (default: 0)
          --spectral (int) : Extract Fourier transform features or not (default: 0)
          --normalized (int) : normalized data by channel (default: 0)
          --setup (str) : "eegnet" or "erpenet" => "erpenet" returns different output format (default: "eegnet")
```

Please, in case you choose to reproduce feature extraction step please make sure the following args :
- name_feature : **time_domain**, ica : 0, spectral : 0 (ica_time_domain if ica : 1)
- name_feature : **spectral_domain**, ica : 0, spectral : 1 (ica_spectral_domain if ica : 1)
- name_feature : **erpenet_time_domain**, ica : 0, spectral : 0, setup : "erpenet"
- name_feature : **erpenet_time_normalized**, ica : 0, spectral : 0, normalized : 1, setup : "erpenet"
    

### Or download the different extracted features :
```shell script
$ mkdir inria-bci-challenge
$ cd inria-bci-challenge
$ wget "https://www.dropbox.com/s/sodpx5l2i4tyoug/features.tar.gz"
$ tar -xzvf features.tar.gz
$ rm features.tar.gz
```


## First approach : fit models on the extracted features


## Second approach : Train Neural Networks on raw data



## References







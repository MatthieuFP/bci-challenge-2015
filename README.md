## Introduction

This project aims at completing the Kaggle BCI Challenge @ NER 2015 : https://www.kaggle.com/c/inria-bci-challenge/overview

It consists in discriminating wrong from good feedbacks from a P300 Speller experiment. 
I used two different approaches in this project and compare them :
1) Extracting relevant and then fit machine learning models
2) Training deep neural networks (EEGNet and ERPENet) on raw data

You can vizualise some EEG data with in the notebook vizualisation.ipynb.

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

Fit Logistic regression, ElasticNet, SVM or XGBoost on the different extracted features and returns the .csv prediction file.
Please run one of the following command line to fit a model on the desired features.

```shell script
$ python3 classifier/logistic_reg.py --wavelets 0 --xdawn 0 --save_name "name" --iter 400 --penalty "l2" --solver "lbfgs"
$ python3 classifier/elasticnet.py --wavelets 0 --xdawn 0 --save_name "name" --iter 400 --ica 0
$ python3 classifier/svm.py --wavelets 0 --xdawn 0 --save_name "name" --iter 1000 --C 1. --kernel "rbf"
$ python3 classifier/xgb.py --wavelets 0 --xdawn 0 --save_name "name" --iter 400 --max_depth 6 --n_estimators 200



    options:
          --wavelets (int) : compute wavelets coefficients to use them as features or not (default: 0)
          --xdawn (int) : compute Spatial filters to use them as features or not (default: 0)
          --save_name (str) : name of the returned prediction file on the test set (Required)
          --iter (int) : number of max. iterations to perform (default: 400 or 1000)

      logistic_reg.py :
          --penalty (str) : "l2" or "l1" (default: "l2")
          --solve (str) : solver, please refer to the scikit-learn doc to select different solver (default: "lbfgs")

      elasticnet.py :
          --ica (int) : use ica features to fit model or not (default: 0)

      svm.py :
          --C (float) : penalty coefficient (default: 1.)
          --kernel (str) : kernel, please refer to the scikit-learn doc to select different kernel (default: "rbf")

      xgb.py :
          --max_depth (int) : max depth of the xgboost trees (default: 6)
          --n_estimators (int) : number of estimators (default: 200)
```

## Second approach : Train Neural Networks on raw data

Train EEGNet or ERPENet :

```shell script
$ python3 main.py --model "eegnet" --features "time_domain" --epochs 50 --batch_size 32 --lr 1e-3 --debug 0 --dropout 0.25 --patience 5 --name_saved "name" --gaussian_noise 0.0

    options:
          --model (str) : either "eegnet" or "erpenet" (Required)
          --features (str) : name of the features, please refer to the list below, make sure to use erpenet features with erpenet model RECOMMENDED : "time_domain" (default: "time_domain")
          --epochs (int) : Number of epochs for training (default: 50)
          --batch_size (int) : size of the batch (default: 32)
          --lr (float) : learning rate (default: 1e-3)
          --debug (int) : performs debugging or not (default: 0)
          --dropout (float) : dropout probability (default: 0.25)
          --patience (int) : number of epochs to wait before early stopping (default: 5)
          --name_saved (str) : name of the folder where to save model and predictions on the test set (Required)
          --gaussian_noise (float) : standard deviation of the gaussian noise to perform data augmentation. If 0.0, no data augmentation is performed (default: 0.0)
```

## References

<a id="1">[1]</a> 
Vernon J. Lawhern and Amelia J. Solon and Nicholas R. Waytowich and Stephen M. Gordon and Chou P. Hung and Brent J. Lance,
EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces,
CoRR,
2016

<a id="2">[2]</a> 
Ditthapron, Apiwat and Banluesombatkul, Nannapas and Ketrat, Sombat and Chuangsuwanich, Ekapol and Wilaiprasitporn, Theerawit,
Universal Joint Feature Extraction for P300 EEG Classification Using Multi-Task Autoencoder,
Institute of Electrical and Electronics Engineers (IEEE),
2019

<a id="3">[3]</a> 
Rivet, Bertrand and Souloumiac, Antoine and Attina, Virginie and Gibert, Guillaume,
xDAWN algorithm to enhance evoked potentials: application to brain-computer interface,
IEEE transactions on bio-medical engineering,
2009










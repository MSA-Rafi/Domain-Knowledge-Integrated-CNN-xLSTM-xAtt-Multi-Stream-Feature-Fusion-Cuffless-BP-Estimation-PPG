[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-knowledge-integrated-cnn-xlstm-xatt/blood-pressure-estimation-on-mimic-iii)](https://paperswithcode.com/sota/blood-pressure-estimation-on-mimic-iii?p=domain-knowledge-integrated-cnn-xlstm-xatt)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-knowledge-integrated-cnn-xlstm-xatt/blood-pressure-estimation-on-vitaldb)](https://paperswithcode.com/sota/blood-pressure-estimation-on-vitaldb?p=domain-knowledge-integrated-cnn-xlstm-xatt)

# Domain Knowledge Integrated CNN-xLSTM-xAtt Network with Multi Stream Feature Fusion for Cuffless Blood Pressure Estimation from Photoplethysmography Signals

🔬 This is the code base for paper: [Domain Knowledge Integrated CNN-xLSTM-xAtt Network with Multi Stream Feature Fusion for Cuffless Blood Pressure Estimation from Photoplethysmography Signals](https://kwnsfk27.r.eu-west-1.awstrack.me/L0/https:%2F%2Fauthors.elsevier.com%2Fc%2F1l6MV_LnESZKxl/1/01020196dba1b04f-e32a503b-d6b3-4c59-96fd-3c64076ae0e1-000000/VlXteY78VEhGn9zPa4TQ9A_Mq98=426)

This repository contains the implementation of a novel deep learning framework for cuffless blood pressure (BP) estimation from photoplethysmography (PPG) signals. The architecture combines CNN-xLSTM-xAttention networks with a multi-stream feature fusion network (MSFN) and a domain knowledge-integrated supervision strategy (D-QuEST). Key components include:

- 🎯 Signal preprocessing with peak-based signal selection (**VSS**) and convolutional peak enhancement (**CPE**)
- 🎯 Multi-domain feature extraction from both 1D signals (via **ConvLSTM and xLSTM-based MUS-Net**) and PPG-derived image modalities (via **pretrained InceptionV3-based MDI-Net**)
- 🎯 Multi-stream attention through a novel **M-SCAN** module for spatial and cross-stream focus
- 🎯 Dual stage **multi-stream fusion network (MSFN)** for combining signal and image domain features using CNN and LSTM blocks
- 🎯 Domain knowledge integrated supervision via a custom **dynamic quantitative embedding supervision-based tuning loss (D-QuEST)**

The framework achieves state-of-the-art performance on two public datasets (MIMIC, and PulseDB), demonstrating robust generalization across diverse subjects and significant improvements in MAE, MSE, and PCC metrics.

## 🧠 Abstract

Estimating blood pressure (BP) from photoplethysmography (PPG) signals is challenging due to signal variability and noise, as well as the complex relationship between PPG and BP, which requires sophisticated algorithms and personalization to achieve high accuracy. This paper introduces a novel deep learning (DL)-based framework, which includes a CNN-xLSTM-xAtt-based deep neural network for extracting multi-domain features, and a multi-stream feature fusion network (MSFN) with a signal enhancer for BP estimation from PPG signals. The performance of this framework is further enhanced by incorporating domain knowledge into the DL network via an ML-based loss function. The data preprocessing step incorporates a novel signal selector that accepts signals with peak height and distance variance within a specified range. To ensure robust feature learning from the peaks, a convolution-based peak enhancement (CPE) method has been employed in this work. For feature extraction, the refined signals are processed using two parallel networks, enabling complementary feature learning from both the uni-dimensional signal and image domains. The uni-dimensional path of the proposed framework, containing two parallel networks–ConvLSTM and xLSTM–facilitate both short- and long-term temporal dependencies. Concurrently, the image-based network extracts morphological and spectral features from three different image modalities, utilizing the benefits of pretrained networks. To ensure the framework focuses on the most important source in this multi-stream network, a novel multi-stream spatial- and cross-attention network (M-SCAN) is proposed. Finally, combining both type of features by a CNN-xLSTM-based multi stream fusion network (MSFN) provides an estimation of BP. During the training of the DL network, a domain knowledge-integrated dynamic quantitative embedding supervision-based tuning (D-QuEST) is also proposed as a supervision loss for the model. The performance of the proposed framework has been tested on two publicly available datasets suggesting 0.05% MSE, 1.40% MAE, and 99.36% PCC on the small and 0.06% MSE, 1.58% MAE, and 99.09% PCC on the large dataset. The proposed framework outperforms existing state-of-the-art methods by a significant margin, even with high subject diversity. This demonstrates its potential for accurate BP estimation solely from PPG signals with diverse variations.

## 🔁 Install and Compile the Prerequisites

- python 3.8
- PyTorch >= 1.8
- pywavelets
- neurokit2
- Python packages: numpy, pandas, scipy

## 📁 Code Organization (Under Codes Folder)

The codes of the work as organized as follows:

| File/Module               | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Load_data.py              | Base code for loading preprocessed data                                     |
| utils.py                  | Custom loss functions, PCC (Pearson Correlation Coefficient) calculation    |
| Image_transform.py        | Transforms PPG signals into raw images, scalogram, and MTF representations  |
| Feature_extractor.py      | Handcrafted feature extraction from PPG signals                             |
| D_QuEST.py                | Domain knowledge integration via Dynamic Quantitative Embedding Supervision |
| PPG2BP_Net.py             | Proposed DL architecture (MDI-Net, MUS-Net, MSFN modules)                   |
| Train_PPG2BP_Net.py       | Training and evaluation of the complete BP estimation framework             |

## 📊 Results Summary

| Dataset | MSE (%) | MAE (%) | PCC (%) |
| ------- | ------- | ------- | ------- |
| Small   | 0.05    | 1.40    | 99.36   |
| Large   | 0.06    | 1.58    | 99.09   |

## 📌 Citation

If you find this work useful, please cite using:

```
@article{SHOAIBAKHTERRAFI2025127994,
title = {Domain knowledge integrated CNN-xLSTM-xAtt network with multi stream feature fusion for cuffless blood pressure estimation from photoplethysmography signals},
journal = {Expert Systems with Applications},
volume = {286},
pages = {127994},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.127994},
url = {https://www.sciencedirect.com/science/article/pii/S095741742501615X},
author = {Md {Shoaib Akhter Rafi} and Md Kamrul Hasan},
keywords = {Blood pressure, Photoplethysmography signals, CNN-xLSTM-xAtt, Quantitative embedding supervision, Multi-stream cross-attention, Multi-stream fusion network}
}
```

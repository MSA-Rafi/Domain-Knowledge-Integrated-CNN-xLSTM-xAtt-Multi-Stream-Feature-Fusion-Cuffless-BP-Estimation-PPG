# Domain Knowledge Integrated CNN-xLSTM-xAtt Network with Multi Stream Feature Fusion for Cuffless Blood Pressure Estimation from Photoplethysmography Signals

This is the base code for paper [Domain Knowledge Integrated CNN-xLSTM-xAtt Network with Multi Stream Feature Fusion for Cuffless Blood Pressure Estimation from Photoplethysmography Signals](https://doi.org/10.1016/j.eswa.2025.127994)

## Abstract

Abstract: Estimating blood pressure (BP) from photoplethysmography (PPG) signals is challenging due to signal variability and noise, as well as the complex relationship between PPG and BP, which requires sophisticated algorithms and personalization to achieve high accuracy. This paper introduces a novel deep learning (DL)-based framework, which includes a CNN-xLSTM-xAtt-based deep neural network for extracting multi-domain features, and a multi-stream feature fusion network (MSFN) with a signal enhancer for BP estimation from PPG signals. The performance of this framework is further enhanced by incorporating domain knowledge into the DL network via an ML-based loss function. The data preprocessing step incorporates a novel signal selector that accepts signals with peak height and distance variance within a specified range. To ensure robust feature learning from the peaks, a convolution-based peak enhancement (CPE) method has been employed in this work. For feature extraction, the refined signals are processed using two parallel networks, enabling complementary feature learning from both the uni-dimensional signal and image domains. The uni-dimensional path of the proposed framework, containing two parallel networks–ConvLSTM and xLSTM–facilitate both short- and long-term temporal dependencies. Concurrently, the image-based network extracts morphological and spectral features from three different image modalities, utilizing the benefits of pretrained networks. To ensure the framework focuses on the most important source in this multi-stream network, a novel multi-stream spatial- and cross-attention network (M-SCAN) is proposed. Finally, combining both type of features by a CNN-xLSTM-based multi stream fusion network (MSFN) provides an estimation of BP. During the training of the DL network, a domain knowledge-integrated dynamic quantitative embedding supervision-based tuning (D-QuEST) is also proposed as a supervision loss for the model. The performance of the proposed framework has been tested on two publicly available datasets suggesting 0.05% MSE, 1.40% MAE, and 99.36% PCC on the small and 0.06% MSE, 1.58% MAE, and 99.09% PCC on the large dataset. The proposed framework outperforms existing state-of-the-art methods by a significant margin, even with high subject diversity. This demonstrates its potential for accurate BP estimation solely from PPG signals with diverse variations.

## Install and compile the prerequisites

- python 3.8
- PyTorch >= 1.8
- pywavelets
- neurokit2
- Python packages: numpy, pandas, scipy

## Code Organization

The codes of the work as organized as follows:
- Load_data.py -> Base code for loading preprocessed data
- utils.py -> Custom loss functions, PCC calculation
- Image_transform.py -> Transforming PPG signals to raw images, scalogram, and MTF images
- Feature_extractor.py -> Handcrafted feature extraction from PPG signals
- D_QuEST.py -> Proposed method for domain knowledge integration into the DL model through dynamic quantitative embedding supervision-based tuning
- PPG2BP_Net.py -> Proposed architecture of the DL model with MDI-Net, MUS-Net, and MSFN
- Train_PPG2BP_Net.py -> Training and evaluation code for the proposed framework

## Citation

If you find this work helpful, please cite using:

``@article{shoaib2025domain,
  title={Domain Knowledge Integrated CNN-xLSTM-xAtt Network with Multi Stream Feature Fusion for Cuffless Blood Pressure Estimation from Photoplethysmography Signals},
  author={Shoaib, Md and Rafi, Akhter and Hasan, Md Kamrul},
  journal={Expert Systems with Applications},
  pages={127994},
  year={2025},
  publisher={Elsevier}
}``

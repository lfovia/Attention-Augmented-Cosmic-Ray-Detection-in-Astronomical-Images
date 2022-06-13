# Attention-Augmented-Cosmic-Ray-Detection-in-Astronomical-Images
For any queries, please contact at ee19resch01008@iith.ac.in


# Abstract
Cosmic Ray (CR) hits are the major contaminants in astronomical imaging and spectroscopic observations involving solid-state detectors. Correctly identifying and masking them is a crucial part of the  image processing pipeline, since it may otherwise lead to spurious detections. For this purpose, we have developed and tested a novel Deep Learning based framework for the automatic detection of CR hits from astronomical imaging data from the Dark Energy Camera (DECam) observations. We considered two baseline models namely deepCR and Cosmic-CoNN, which are the current state-of-the-art learning based algorithms that were trained using Hubble Space Telescope (HST) ACS/WFC and LCOGT Network images respectively. We have experimented with the idea of augmenting the baseline models using Attention Gates (AGs) to improve the CR detection performance. We have trained our models on DECam data and demonstrate a consistent marginal improvement by adding AGs in True Positive Rate (TPR) at 0.01\% False Positive Rate (FPR) and Precision at 95\% TPR over the aforementioned baseline models for the DECam data set. Furthermore, we demonstrate that the proposed baseline models with and without attention augmentation outperform state-of-the-art models such as Astro-SCRAPPY, Maximask (that is trained natively on DECam data) and pre-trained ground-based Cosmic-CoNN. This study demonstrates that the AG module augmentation enables us to get a better deepCR and Cosmic-CoNN models and to improve their generalization capability on unseen data. 

# Acknowledgment
We considered our codes mainly from the following sources.
1. https://github.com/profjsb/deepCR (deepCR)
2. https://github.com/sfczekalski/attention_unet (Attention U-Net)

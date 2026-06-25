# MLMLMLM

## Muscle-Listening Machine Learning Model for Live Music


Lucy Strauss, Prashanth Thattai Ravikumar, and Matthew Yee-King. 2026. Cross-Modal Sig2Sig Machine Translation with Deep Generative Modeling for NIME Design. Proceedings of the International Conference on New Interfaces for Musical Expression. DOI: [10.5281/zenodo.20784411](https://zenodo.org/records/20784411)


MLMLMLM is a custom model architecture for translating electromyographic (EMG) signals into probable audio in live music performance settings. This is implemented through EMG-conditioned sequence generation of audio signals.


The MLMLMLM model architecture is composed of two RVQ-VAEs and a decoder-only Transformer. Each model is trained separately. One RVQ-VAE models EMG signals and the other RVQ-VAE models audio signals. The Transformer is implemented in latent space, using quantized latent vectors of the audio RVQ-VAE for self-attention, and quantized latent vectors of the EMG RVQ-VAE for cross-attention.


## Requirements



### Training Requirements


[stable-audio-tools](https://github.com/stability-ai/stable-audio-tools)


### Inference Requirements


[pybela](https://github.com/BelaPlatform/pybela)


### Additional Requiremets


See `setup.py` for additional requirements.


To run training, you will also need a time-aligned dataset of audio and EMG signals. We have not made our dataset public because we consider this to be part of the artwork and the musician did not wish to make the dataset publicly available. However, you can reach out to us [here](https://lucystrauss.com/contact). We share this repo for research purposes, but do not intend for our exact model training to be reproducible. Take the script, remix it and make your own version! Please cite the repo and [paper](https://nime.org/proc/nime2026_133/index.html) if you do.




# Install

First, install `stable-audio-tools` from PyPI with:
```bash
$ pip install stable-audio-tools
```

Then:
```bash
$ pip install .
```



# Shoutouts


The scripts for live interaction in performance settings are built on [pybela](https://github.com/BelaPlatform/pybela).



The models in this repo are adapted from [stable-audio-tools](https://github.com/stability-ai/stable-audio-tools). Our main changes are to adapt the model architecture capacity and hyperparameters for deep generative modeling of a 6-channel EMG dataset. Additionally, we added functionality to training scripts to allow for causal, autoregressive sequence generation with streaming conditioning, necessary for live performance scenarios. Our training procedure and architecture composition is also different. For full details, check out our paper.



# Please Cite:



Lucy Strauss, Prashanth Thattai Ravikumar, and Matthew Yee-King. 2026. Cross-Modal Sig2Sig Machine Translation with Deep Generative Modeling for NIME Design. Proceedings of the International Conference on New Interfaces for Musical Expression. DOI: [10.5281/zenodo.20784411](https://zenodo.org/records/20784411)



## BibTeX Entry:



```BibTeX
@inproceedings{nime2026_133,
abstract = {},
address = {London, United Kingdom},
articleno = {133},
author = {Lucy Strauss and Prashanth Thattai Ravikumar and Matthew Yee-King},
booktitle = {Proceedings of the International Conference on New Interfaces for Musical Expression},
doi = {10.5281/zenodo.20784411},
editor = {Benedict Gaster and João Tragtenberg and Anna Xambó and Tom Mitchell},
issn = {2220-4806},
month = {June},
numpages = {18},
pages = {1084--1101},
presentation-video = {https://youtu.be/Z7-ySfuF7lg},
title = {Cross-Modal Sig2Sig Machine Translation with Deep Generative Modeling for NIME Design},
track = {paper},
url = {http://nime.org/proceedings/2026/nime2026_133.pdf},
year = {2026}
}
```


## TODO:

- [ ] train instructions
- [ ] run instructions

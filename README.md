# StyleGAN2-ADA &mdash; for Audio Textures 

This forked repository has updates for modelling Audio Textures. Please see the original official README from NVIDIA [here](https://github.com/NVlabs/stylegan2-ada-pytorch) for details on licenses and citations.

## Compatibility
Note: This version of StyleGAN2 is not compatible with PyTorch>1.8. We use PyTorch 1.7 for all experiments.

## Setup
* Clone this repo
* Install dependencies by creating a new conda environment called ```audio-stylegan2``` (Note: if you are using this repo in conjunction with [audio-latent-composition](https://github.com/pkamath2/audio-latent-composition), you do not need to recreate this. Both projects need the same environment setup)
```
conda env create -f environment.yml
```
Add the newly created environment to Jupyter Notebooks
```
python -m ipykernel install --user --name audio-stylegan2
```

## Training
Kickstart training using the command below. See [config.json](config/config.json) for various parameter settings.  

Note this is unsupervised training. 
TODO: Integrate conditional training from original repo.
```
python main.py --data_dir=<data location> --out_dir=training-runs/<checkpoint location>
```

## Notebooks
Notebooks outline how to generate randomly from trained GAN. Further, we use Phase Gradient Heap Integration (PGHI) method to invert Spectrograms to audio. See (this)[https://ieeexplore.ieee.org/abstract/document/7890450/] paper and (this)[https://arxiv.org/pdf/2103.07390.pdf] paper for details. StyleGAN architectures for audio learn spectrogram representations as images and thus need to be scaled from [0,255] to [-50,0].

* [Random generation](notebooks/generate.ipynb)


## Semantic Factorization
We use the SeFa algorithm from (this)[https://genforce.github.io/sefa/] paper to automatically find directions for controllability. 

TODO: Add the notebook and streamlit interface


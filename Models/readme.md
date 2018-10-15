# Conditional GANs for the Reconstruction of 3D Anatomical Structures from 2D Planes

We implemented a conditional Generative Adversarial Networks (cGAN) using Keras and tensorflow backend to reconstruct the 3D structure of a fetal skull from its correspodning 2D views.
We also implemented a different model for the reconstruction of the 3D structure of the left ventricle of the heart using its 2D plane views.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python 3.6.5
* Keras
* Tensorflow
* Numpy
* Nibabel

### Training

To train the models, you will need to specify which implementation: the fetal or cardiac model is needed.

For fetal:

```
cd fetal
python cGAN_split_iter.py
```

For cardiac:

```
cd cardiac
python train.py
```

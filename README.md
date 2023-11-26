# PoInt-Net 

## News


![pipeline](Front_img.png)

## Requirements

- Python 3.7+
- Pytorch
- numpy
- scipy
- scikit-learn
- scikit-image
- MiDaS

## Data Preparation

**With depth:**

- Run depth2normal.py to precompute the surface normal.
- Run img2pcd.py to preprocess the rgb-d image to point cloud.

**Without depth:**

- Go to [MiDaS](https://github.com/isl-org/MiDaS) to precompute the depth （or any other depth estimation model）
- Run depth2normal.py to precompute the surface normal.
- Run img2pcd.py to preprocess the rgb-d image to point cloud.

## Evaluation

- Run 'test.py' to decompose the albedo and shading. (We provide example data in folder: Data)

## Pre-trained Model

We provide pre-trained models in the pre_trained_model folder.




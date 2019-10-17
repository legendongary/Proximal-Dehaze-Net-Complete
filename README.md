# Proximal-Dehaze-Net-Complete

MATLAB implementation of ECCV 2018 paper "Proximal Dehaze-Net: A Prior Learning-Based Deep Network for Single Image Dehazing" with several improvements. We provide the full functions of training and evaluation as well as our trained network models.

#### Installation

This work is implemented based on [MatConvNet](http://www.vlfeat.org/matconvnet/) package and we have included Linux pre-compiled files in this project. To train our network, we require a good GPU device and CUDA toolkit.

#### Evaluation

We provide several trained models for directly network evaluation. The models are located in `./models/`. For an example of using these models for image dehazing as shown in :

```matlab
% use network with one stage trained on our dataset
% image: input hazy image
% resim: recovered haze-free image
% restt: estimated transmission
[resim, restt] = ours_tiphqs_s1_eval(image)

% use network with two stage trained on RESIDE dataset
% image: input hazy image
% resim: recovered haze-free image
% restt: estimated transmission
[resim, restt] = ours_tipres_s2_eval(image)
```

#### Training

To train our network, the training dataset must be generated first. We offer the training images and code to generate training data. Please download from [Baiduyun](https://pan.baidu.com/s/1DOBd_rJW8Owz-km5mQNySQ). After generating training data, move them to `./data/train/` or modify `opts.imdbPath` in each training file to the data path.

Run following code will then train all networks:

```matlab
run_train
```

#### To be continued ...

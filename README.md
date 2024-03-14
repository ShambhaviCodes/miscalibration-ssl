<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

### Prerequisites
This repo is built on top of [USB](https://github.com/microsoft/Semi-supervised-learning/tree/main).
USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

From now on, you can start use USB by typing 

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
```

### Running the Calibration Method 
You can modify the config files to add the two parameters, margin and the weight to the penalty term.
For example,

```sh
margin_hyperparam: 10
p_margin: 0.1
```

Alternatively, you can run with a modified config file :

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0_penalty.yaml
```

To evaluate the model for ECE and Errors :

```sh
python eval.py --dataset cifar100 --num_classes 100 --load_path ./saved_model/best_model.pth
```

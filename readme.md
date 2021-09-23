# Classification

Abstract: this repo includes a pipeline using PyTorch with Catalyst for training model for STL-10 image classification.
Weights for trained models are provided.

## Plan of research and general thoughts
The main idea to go through different backbones and classifiers and to find the right combination that can be fine-tuned the best.
ResNet50 did the best job as a backbone (pretrained on ImageNet), but I haven't tried EfficientNet family for this task, the results might be better.
The classifier can be checked in model class.

Images are resized to standart (224, 224), but nother approach can be to modify first convolution layer of backbone or add a new one. 
Generally speaking, this way the distribution can get different from ImageNet. But the model converged good even in this case.

I have not used unlabeled data mainly due to limited time. 
However, I'm going to **describe** several approaches to use it. 
I've conducted a research on modern SOTA semi-supervised approaches (SSL) for such tasks.

There are some really cool ideas like https://paperswithcode.com/paper/spinalnet-deep-neural-network-with-gradual-1

Quite intuitive approach can be to train classifier first on labeled data and then to pseudo-classify unlabaled data using thershold. For classifying unlabaled data we can use weak augmentation (for better results) and after retrain our network using strong augmentation of the new dataset.
https://arxiv.org/abs/2001.07685

Another paper suggests similar approach, but using dynamic thershold while pseudo-labeling unlabeled data.
http://proceedings.mlr.press/v139/xu21e/xu21e.pdf

But I believe the metrics can increase in 1-2% this way in this case, mainly because provided dataset contains two times more labeled data then the original STL-10, but these are good approaches for SSL.

By the way, I'm providing **EDA** to check the similarity between train and test data distributions.
Classes are said to be balanced, that's good.

To sum up:
 - Architecture: ResNet50, MobileNetV2 with custom heads
 - Several-stage fine-tuning
 - Loss function: BinaryCrossEntropy
 - Optimizer: SGD (learning rate for backbone 1e-3, learning rate for classifier 1e-2, momentum 0.9), Adam didn't work good for this case
 - learning scheduler: ReduceLROnPlateau(factor=0.15, patience=2)
 - EarlyStopping(patience=5, min_delta=0.001)

## Results

| Backbone | Accuracy (test) | Averaged F1 (test) | Loss | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| MobilenetV2 | 0.4132 |     |     |      |    |
| ResNet-50  | 0.96  |     |     |   35     |

The fine-tuning has been done in several stages, starting from training only classifier for several epochs and then slightly unfreezing the backbone layers (including unfreezing BatchNorm).

P.S. If we check provided EDA, we can notice that in the test case there are less outliers in comparison to train data. Thus, for test data we have some "easier" distribution.

**Link to TensorBoard for ResNet50:** [tap here](https://tensorboard.dev/experiment/rTq70zmmRJeXbklyFQs46g/#scalars)

## Installation

Catalyst library is required [catalyst](https://nodejs.org/).

Installation:

```sh
pip install catalyst
```
# Usage

**The directory tree should be:**

<pre>
├── Predict_masks.py
├── Train.py
├── config.py
├── data
│   ├── test_images            #download test images here
│   └── train_images           #download train images here
├── images
├── readme.md
├── utils
│   └── utils.py
└── weights
    ├── UnetEfficientNetB4_IoU_059.pth
    └── UnetResNet50_IoU_043.pth 
</pre>

## Evaluation

There is a Predict_classes.py script which can be used to evaluate the model and predict classes for each image in the test directory. The weights should be stored in the ./weights directory.

Usage example:

```sh
python3 Predict_classes.py -dir /Users/user/Documents/steel_defect_detection/data/  -weights_dir /Users/user/Documents/steel_defect_detection/data/weights
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "train" and "train.csv".
-weights_dir   : Pass a weights directory.
```

## Training

The model is supposed to be trained on the STL-10 dataset. 
You can choose which backbone to use and a batch size. The default is ResNet50.

It is necessary to point the directory where the train folder is stored.

Usage example:
```sh
python3 Train.py -dir /Users/user/Documents/steel_defect_detection/data/ -num_of_workers 4
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "train".
-encoder   : Backbone to use as encoder, default='resnet50'.
-batch_size   : Batch size for training, default=8.
-num_of_workers   : Number of workers for training, default=0.
```
[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
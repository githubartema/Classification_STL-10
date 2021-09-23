# Classification

Abstract: this repo includes a pipeline using PyTorch with Catalyst for training model for STL-10 image classification.
Weights for trained models are provided.

## Plan of research and general thoughts
The main idea to go through different backbones and classifiers and to find the right combination that can be fine-tuned the best.
I've compared ResNet50 and MobileNetV2 as backbones (pretrained on ImageNet).
The classifier can be checked in model class.

P.S. I haven't tried EfficientNet family for this task, the results might be better.

Images are resized to standart (224, 224). I'm not sure that's the best way to deal with it, but should be better than to rely on using average adaptive pulling as 
kernels might not be able to extract the features with the usual size.

I have not used unlabeled data mainly due to limited time. 
However, I'm going to **describe** several approaches to use it. 
I've conducted a research on modern SOTA semi-supervised approaches (SSL) for such tasks.

There are some really cool ideas like https://arxiv.org/pdf/1911.09265v2.pdf.

The more intuitive approach can be to train classifier first on labeled data and then to pseudo-classify unlabeled data using threshold. For classifying unlabaled data we can use weak augmentation (for better results) and after retrain our network using strong augmentation for the new dataset.
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
 - Optimizer: SGD (learning rate for backbone 1e-3, learning rate for classifier 1e-2, momentum 0.9), Adam didn't work good for this case.
 - learning scheduler: ReduceLROnPlateau(factor=0.15, patience=2)
 - EarlyStopping(patience=5, min_delta=0.001)

## Results

| Backbone | Accuracy (test) | Averaged F1 (valid) | Loss | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| MobilenetV2 | 90% |  89%  |  1.56   |   28   |    
| ResNet-50  | 96%  |  96%  |  1.5  |   35     |

The fine-tuning has been done in several stages, starting from training only classifier for several epochs and then slightly unfreezing the backbone layers (including unfreezing BatchNorm). Network class alows to do that.

P.S. If we check provided EDA, we can notice that in the test case there are less outliers in comparison to train data. Thus, for test data we have some "easier" distribution.

**Link to TensorBoard for ResNet50:** [tap here](https://tensorboard.dev/experiment/SmExjs0QTeuNPDDDltnfNg/#scalars&_smoothingWeight=0&runSelectionState=eyJ0cmFpbiI6dHJ1ZSwiX2Vwb2NoXyI6dHJ1ZX0%3D)

## Installation

Catalyst library is required [catalyst](https://github.com/catalyst-team/catalyst).

Installation:

```sh
pip install catalyst
```
# Usage

**The directory tree should be:**

<pre>
.
├── EDA_Classification.ipynb
├── Predict_classes.py
├── Train.py
├── config.py
├── data
│   ├── test
│   │   └── images
│   └── train
│       ├── airplane
│       ├── bird
│       ├── car
│       ├── cat
│       ├── deer
│       ├── dog
│       ├── horse
│       ├── monkey
│       ├── ship
│       └── truck
├── model
│   └── finetune_model.py
├── readme.md
├── utils
│   └── utils.py
└── weights
    └── best-last-resnet50.pth
</pre>

## Evaluation

There is a Predict_classes.py script which can be used to evaluate the model and predict classes for each image in the test directory. The weights should be stored in the ./weights directory.

Usage example:

```sh
python3 Predict_classes.py -dir /Users/user/Documents/Classification_STL-10/data/  -weights_dir /Users/user/Documents/Classification_STL-10/weights
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "test/images".
-weights_dir   : Pass a weights directory.
```

## Training

The model is supposed to be trained on the STL-10 dataset. 
You can choose a batch size to use. 

It is necessary to point the directory where the train folder is stored.

Usage example:
```sh
python3 Train.py -dir /Users/user/Documents/Classification_STL-10/data/ -num_of_workers 4
```
### Arguments
```sh
-dir    : Pass the full path of a directory containing a folder "train".
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

# Computer Vision: Identifying Pnumonia From Chest X-Rays
### Michael Mahoney
##### Date: 12/11/2020

# Abstract
This notebook will attempt to create and analyze a neural network that is effective at identifying pneumonia in patient X-rays. The dataset used in this analysis was pulled from Kaggel, an open source data science competition website. We will build a convulutional neural network with two goals in mind. As a tool to be utilized in the medical industry the primary motivation and purpose for the model will be to identify sick patients. With this in mind, maximizing the positive case recall will be essentail for any model that's production worthy. 

The secondary, but also important, fuction of the model is to be accurate. Screening is significantly less useful if the underlying system is inundated with false-positive cases. While identifying patients with pneumonia is essentail, maximizing the model's recall score can't require abandoning the ability to distinguish which patients are healthy.
This notebook follows the guidelines of the CRISP-DM method of doing data-science. We will begin with a quick overview of the problem at hand, understand the challenges of classifying images, set a beseline and iteratively improve upon the baseline. The notebook will wrap up with a second benchmark, testing the arcitecture in this paper against a tuned VGG19 open source model imported for transfer learning. The notebook will conclude with an overal analysis of the final model's performace and relate it back to the industry. 

The model built in this paper was effective at classifying patients with pnumonia. The best iteration of the model had an overall accuracy of 93% and was correctly predicting 97% of all patients that had pneumonia and 88% of non-pneumonia patients. 
# Bussiness Understanding
Often times in the medical industry, images are a window into the complicated world of diagnoses. Anyone reading this is likely somewhat familiar with the standard workflow for diagnostic image processing. For completeness we will break this down into the emergancy and non-emergancy cases

1. Emergancy: 

    A patient comes into an emergancy room with respritory illness symptoms. They are sent for medical x-rays/ultasounds and the images are quickly reviewed by a radiologist while the patient is stabalized. The radiologist will then share any diagnosis with the patients emergancy room doctor and care will continue from there. 


2. Non-emergancy: 

    Anyone reading this paper is likely more familiar with this process. A patient comes to an medical imaging office with some respritory related issue (coughing, wheezing, difficulty breathing). They describe their symptoms and to a PA/MD and are taken for medical imaging. The x-rays/ultrasounds are generated on site but are then transmitted to a radiologist (contractor that is commonly not on site or on call at all times). In the mean time, the patient, if not showing any emergancy symptoms, is sent home with minor medication. Sometime later, The Radiologist reviews the images as they come and reports back with their diagnosis to the non-emergancy care team who then contacts the patient about further care. 

Where will our model have impact? Primarily, the model can function as an effective triage for non-emergancy patients. During the time in after the image is taken and before the radiologist is able to review them, patients are vulnerable to the positive conditions that worsen and become critical prior to the positive diagnosis. At this point, the patient is forced to go to the emergancy room and seek immediate medical attention. With the patient's health and finances on the line there are very tangible benefits to the patient by correctly identifying illness before the conditions worsens to emergancy conditions. Moreover, it corrects the issue at a non-emergancy level which is better financially for the medical institutions since emergancy care is much less likely to pay reliably (insurance and willingness to pay consideration.) A supplimental benefit could be during the emergancy phase of care. Models with high accuracy can be used as an additional check with radiologists beinging attention to tricky diagnoses. The implementation I can foresee in this case is generating a flag or an additional review if the doctor's diagnosis contradicts the model's prediction. 

At this point I think it's important to achknowledge the potential to expand this implemenation to other similar models tuned for other diseases (or potentially a large network with transfer learning). I would ask the reader to keep this in mind as we analyze our specific model because the potential to train similar models for other imaging tasks would be simple and desired. 

# Google Colab details:
If you would like to run this notebook in google colab, the following cells are for mounting and downloading the dataset into the working directory within colab. Mounting your personal drive to colab is easy but does require authentication. Run the cells and read the outputs for the directions. After your drive has been mounted please make sure you have 2.5 gigs available for the images in your colab storage.


```python
# from google.colab import drive
# drive.mount('/content/drive')
```


```python
 
# ## In Python
# import gdown
# url = "https://drive.google.com/uc?id=1Qog9htI2IKJQUxJCsVENEiiuR1hFV_i6"
# output = 'repo.zip'
# gdown.download(url, output, quiet=False)
```


```python
# ! unzip repo.zip;
```

# Package Dependancies
Open this section to view all package and sub dependancies used in this project. If you run this cell and get an import error please pip install the requirements before attempting to proceed. See google python and pip installation for more information.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from sklearn.utils import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import BalancedBatchGenerator
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Sequential, Input, Model, layers, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import tensorflow.keras.metrics as km
from IPython.display import Image
from PIL import Image as pilImage
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import os
import seaborn as sns
```

    /usr/local/lib/python3.6/dist-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.utils.testing module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.utils. Anything that cannot be imported from sklearn.utils is now part of the private API.
      warnings.warn(message, FutureWarning)
    

# Data Understanding
This project is all about x-rays. Loading in a couple different images we see the images are composed of various sizes and pixel brightnesses. There is existing class imbalance in this data of about 3:1 pneumonia to non-pneumonia. In total there are approximately 6000 images which we will be working with. Here are a couple Example images to get you oriented. The first image is the non-pneumonia case. The second is the pneumonia case. 


<img src = 'normal-image1.jpg'>
<img src = 'pneumonia-image-1.jpg'>


# The Baseline Model
For the baseline model, this notebook will showcase the general method and explanations for how the neural network is being created. After this first round, several of the coding details will be left out or trimmed down. Rest assured the logic and decision making will still be expounded upon. 

Rather than using a truly dumb model to start off, we will build a very simpe perceptron with only one dense hidden layer. To keep the baseline model straight forward the dense layer will have 256 nodes that correspond to the dimension of the images it will be attempting to classify. There isn't any general sense that this methodology will yield better results, but it will suffice for a starting point for further iterations.

The hidden layer will utilize the "sigmoid" activation function which should be more appropriate for our scaled images, in which each pixel of information lies between the values of 0 and 1. 

The seaching parameters include stocastic gradient decent as the optimizing function, binary_crossentropy as the loss function and various metrics which we will use later on. Srocastic gradient decent (SGD) is the standard optimizer unless we have specific information which would drive us to other methods. As is true for the loss function. 
## Baseline Model Evaluation
### Evaluation Metrics
As pnumonia is a medical issue we will be using more metrics than usual when evaluating our models. This notebook will contain the following metrics when evaluating models. The Exception to this rule is the baseline which only features the first two metrics because the model is purposefully not optimized. 
* Loss
* Accuracy
* Hinge
* Area Under the Curve (AUC)
* True Positives
* True Negatives
### Accuracy
 For general interpretability, we include accuracy. Accuracy is the metric that best gives an answer to the question, "How does my model compare to an average guess." With this problem in particular, the data has a class imbalance so the question is more, "How does my model compare to a weighted average guess." This suggests that if we weighted a random guess to the proportions of the samples we've been given, we should expect around 74% accuracy on average. In the following two cells, the model is evaluated over the train and test sets, to produce the metrics we are interested in exploring further. 
 
 <img src = 'baseline-model-training-metrics.png'>
 
The graphs above could represent several issues with the model. For the image on the left, the loss function for the validation doesn't change on average over the 5 epochs, while the training loss continues downward roughly exponentially. The discrepancy is very likely do to overfitting of the training data. 

For the image on the right we see the training accuracy increasing logrithmicaly and the validation set ocillating around 75% accuracy. This is very likely do the simplicity of our model not being able to capture useful information about the images in the two dense layers in the baseline model. This is why we see the average accuracy very near what we would expect with a blind guess. The information the model picked up from the training set was particular enough to generalize to images it had never seen before.

For the following models, we will take a much deeper dive into model metrics (as well as indroduce several more). Considering this model is barely treding above a random guess, we will omit a more surgical analysis and move on to the next model. 

## Code To Create The Baseline Model


```python
trainPath = 'dsc-mod-4-project-v2-1-onl01-dtsc-pt-041320/data/chest_xray/train/'
testPath ='dsc-mod-4-project-v2-1-onl01-dtsc-pt-041320/data/chest_xray/test/'
 
# testPath = '/content/drive/MyDrive/Colab Notebooks/flatiron school/dsc-mod-4-project-v2-1-onl01-dtsc-pt-041320/data/chest_xray/test'
# trainPath = '/content/drive/MyDrive/Colab Notebooks/flatiron school/dsc-mod-4-project-v2-1-onl01-dtsc-pt-041320/data/chest_xray/train'
```


```python
# Create an image generator for the training set. We are spliting a validation set off of this training set
trainGenerator0 = ImageDataGenerator(
    rescale = 1./255
)
 
# Create an image generator for the training set. We are spliting a validation set off of this training set
testGenerator0 = ImageDataGenerator(
    validation_split=.3,
    rescale = 1./255
)
```


```python
# The flowed data from the directory for the training set. This will be fed directly into the neural networks
train0 = trainGenerator0.flow_from_directory(
    trainPath,  
    class_mode='binary',
    target_size=(256,256),
    batch_size = 5216,
    shuffle = True)
 
# The split off validation set. This is fed directly into the neural networks
val0 = testGenerator0.flow_from_directory(
    testPath, 
    class_mode='binary',
    target_size=(256,256),
    batch_size = 187,
    subset='validation',
    shuffle = True
)
 
# The flowed data from the directory for the testing set. This will be fed directly into the neural networks
test0 = testGenerator0.flow_from_directory(
    testPath, 
    class_mode='binary',
    target_size=(256,256),
    batch_size = 437,
    subset = 'training',
    shuffle = True
)
```

    Found 5216 images belonging to 2 classes.
    Found 187 images belonging to 2 classes.
    Found 437 images belonging to 2 classes.
    


```python
trainImages, trainImageLabels = next(train0)
valImages, valImageLabels = next(val0)
testImages, testImageLabels = next(test0)
```


```python
weights = compute_class_weight(
    'balanced',
    classes=[0, 1],
    y = train0.labels
)
weights = {0:weights[0], 1:weights[1]}
weights
```




    {0: 1.9448173005219984, 1: 0.6730322580645162}




```python
flatTrainImages = trainImages.reshape(5216,-1)
flatValImages = valImages.reshape(187 ,-1)
flatTestImages = testImages.reshape(437 ,-1)

# Save the shape of a flattened image in order to feed it into the model
inputShape = flatTrainImages.shape[1]
```


```python
# We will be using the keras functional API for building models. 

# define the input layer
baselineModelInputs = Input((inputShape,), name = 'input_layer')

# define the first layer
x1 = layers.Dense(128, activation = 'sigmoid')(baselineModelInputs)

# define the ourput layer
baselineModelOutput = layers.Dense(1, activation = 'sigmoid')(x1)

# Create an instance of the keras.Model class with the input and outputs above
baselineModel = Model(inputs=baselineModelInputs, outputs = baselineModelOutput, name = 'baseline_model')

#compile the model
baselineModel.compile(
    optimizer = 'sgd',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```


```python
# create the file path for where we want to save the resulting model
baseline_filepath = os.path.join('baseline_model')
```


```python
baselineModel.summary()
```

    Model: "baseline_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_layer (InputLayer)     [(None, 196608)]          0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               25165952  
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 25,166,081
    Trainable params: 25,166,081
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# We fit the model, throwing in some basic parameters that will be tuned later. 
baselineHistory = baselineModel.fit(
    flatTrainImages,
    trainImageLabels,
    batch_size=32,
    epochs=100,
    validation_data=(flatValImages, valImageLabels),
    class_weight= weights,
#     callbacks=[baselineEarlyStop, baselineModelCheckpoint]  
)
```

    Epoch 1/100
    163/163 [==============================] - 2s 12ms/step - loss: 0.5763 - accuracy: 0.7088 - val_loss: 0.6773 - val_accuracy: 0.5829
    Epoch 2/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.3658 - accuracy: 0.8457 - val_loss: 0.3687 - val_accuracy: 0.8021
    Epoch 3/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.2881 - accuracy: 0.8811 - val_loss: 0.2740 - val_accuracy: 0.8663
    Epoch 4/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.2286 - accuracy: 0.9024 - val_loss: 0.2309 - val_accuracy: 0.8824
    Epoch 5/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.2112 - accuracy: 0.9156 - val_loss: 0.2269 - val_accuracy: 0.9198
    Epoch 6/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.2138 - accuracy: 0.9101 - val_loss: 0.7614 - val_accuracy: 0.6952
    Epoch 7/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1728 - accuracy: 0.9279 - val_loss: 0.2825 - val_accuracy: 0.8877
    Epoch 8/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1922 - accuracy: 0.9245 - val_loss: 0.5077 - val_accuracy: 0.7807
    Epoch 9/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1713 - accuracy: 0.9298 - val_loss: 0.5596 - val_accuracy: 0.7647
    Epoch 10/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1707 - accuracy: 0.9294 - val_loss: 0.6054 - val_accuracy: 0.7807
    Epoch 11/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1647 - accuracy: 0.9312 - val_loss: 0.5406 - val_accuracy: 0.7861
    Epoch 12/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1489 - accuracy: 0.9415 - val_loss: 0.2420 - val_accuracy: 0.8930
    Epoch 13/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1467 - accuracy: 0.9392 - val_loss: 0.7831 - val_accuracy: 0.7487
    Epoch 14/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1446 - accuracy: 0.9406 - val_loss: 0.8761 - val_accuracy: 0.7273
    Epoch 15/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1380 - accuracy: 0.9427 - val_loss: 0.7611 - val_accuracy: 0.7647
    Epoch 16/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1383 - accuracy: 0.9442 - val_loss: 0.3791 - val_accuracy: 0.8289
    Epoch 17/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1405 - accuracy: 0.9404 - val_loss: 0.4546 - val_accuracy: 0.8289
    Epoch 18/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1413 - accuracy: 0.9419 - val_loss: 0.7195 - val_accuracy: 0.7701
    Epoch 19/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1364 - accuracy: 0.9442 - val_loss: 0.2555 - val_accuracy: 0.8877
    Epoch 20/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1191 - accuracy: 0.9526 - val_loss: 0.7928 - val_accuracy: 0.7540
    Epoch 21/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1283 - accuracy: 0.9490 - val_loss: 0.5172 - val_accuracy: 0.8235
    Epoch 22/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1262 - accuracy: 0.9457 - val_loss: 0.2985 - val_accuracy: 0.8663
    Epoch 23/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1138 - accuracy: 0.9519 - val_loss: 0.8581 - val_accuracy: 0.7326
    Epoch 24/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1146 - accuracy: 0.9532 - val_loss: 0.3882 - val_accuracy: 0.8342
    Epoch 25/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1086 - accuracy: 0.9551 - val_loss: 0.4730 - val_accuracy: 0.8289
    Epoch 26/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1115 - accuracy: 0.9540 - val_loss: 0.4243 - val_accuracy: 0.8289
    Epoch 27/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1087 - accuracy: 0.9565 - val_loss: 0.4596 - val_accuracy: 0.8449
    Epoch 28/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1073 - accuracy: 0.9551 - val_loss: 0.3724 - val_accuracy: 0.8449
    Epoch 29/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0978 - accuracy: 0.9605 - val_loss: 0.8780 - val_accuracy: 0.7219
    Epoch 30/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1088 - accuracy: 0.9557 - val_loss: 0.3613 - val_accuracy: 0.8663
    Epoch 31/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.1026 - accuracy: 0.9584 - val_loss: 0.8173 - val_accuracy: 0.7326
    Epoch 32/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0990 - accuracy: 0.9588 - val_loss: 0.5434 - val_accuracy: 0.8235
    Epoch 33/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0934 - accuracy: 0.9630 - val_loss: 0.5806 - val_accuracy: 0.8075
    Epoch 34/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0948 - accuracy: 0.9603 - val_loss: 0.4938 - val_accuracy: 0.8182
    Epoch 35/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0918 - accuracy: 0.9632 - val_loss: 0.8171 - val_accuracy: 0.7540
    Epoch 36/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0903 - accuracy: 0.9651 - val_loss: 0.4690 - val_accuracy: 0.8182
    Epoch 37/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0915 - accuracy: 0.9634 - val_loss: 0.6148 - val_accuracy: 0.8075
    Epoch 38/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0994 - accuracy: 0.9601 - val_loss: 0.6543 - val_accuracy: 0.7914
    Epoch 39/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0881 - accuracy: 0.9666 - val_loss: 0.8197 - val_accuracy: 0.7433
    Epoch 40/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0842 - accuracy: 0.9701 - val_loss: 0.4814 - val_accuracy: 0.8289
    Epoch 41/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0849 - accuracy: 0.9674 - val_loss: 0.7531 - val_accuracy: 0.7701
    Epoch 42/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0833 - accuracy: 0.9655 - val_loss: 0.3989 - val_accuracy: 0.8610
    Epoch 43/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0860 - accuracy: 0.9676 - val_loss: 0.8200 - val_accuracy: 0.7433
    Epoch 44/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0854 - accuracy: 0.9645 - val_loss: 0.9677 - val_accuracy: 0.7166
    Epoch 45/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0784 - accuracy: 0.9709 - val_loss: 0.5871 - val_accuracy: 0.8182
    Epoch 46/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0789 - accuracy: 0.9712 - val_loss: 0.6217 - val_accuracy: 0.8235
    Epoch 47/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0771 - accuracy: 0.9691 - val_loss: 0.8944 - val_accuracy: 0.7273
    Epoch 48/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0729 - accuracy: 0.9722 - val_loss: 0.4510 - val_accuracy: 0.8396
    Epoch 49/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0740 - accuracy: 0.9699 - val_loss: 1.1025 - val_accuracy: 0.7112
    Epoch 50/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0767 - accuracy: 0.9688 - val_loss: 0.8455 - val_accuracy: 0.7433
    Epoch 51/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0728 - accuracy: 0.9716 - val_loss: 0.7646 - val_accuracy: 0.7754
    Epoch 52/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0669 - accuracy: 0.9739 - val_loss: 1.0268 - val_accuracy: 0.7219
    Epoch 53/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0778 - accuracy: 0.9684 - val_loss: 0.8614 - val_accuracy: 0.7540
    Epoch 54/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0717 - accuracy: 0.9732 - val_loss: 0.6961 - val_accuracy: 0.7968
    Epoch 55/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0655 - accuracy: 0.9749 - val_loss: 1.1578 - val_accuracy: 0.7005
    Epoch 56/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0649 - accuracy: 0.9739 - val_loss: 0.9719 - val_accuracy: 0.7273
    Epoch 57/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0666 - accuracy: 0.9743 - val_loss: 0.9769 - val_accuracy: 0.7326
    Epoch 58/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0645 - accuracy: 0.9747 - val_loss: 0.5755 - val_accuracy: 0.8235
    Epoch 59/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0628 - accuracy: 0.9755 - val_loss: 0.9186 - val_accuracy: 0.7487
    Epoch 60/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0625 - accuracy: 0.9762 - val_loss: 0.6054 - val_accuracy: 0.8021
    Epoch 61/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0626 - accuracy: 0.9762 - val_loss: 1.2281 - val_accuracy: 0.7005
    Epoch 62/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0601 - accuracy: 0.9781 - val_loss: 0.7581 - val_accuracy: 0.7807
    Epoch 63/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0645 - accuracy: 0.9726 - val_loss: 0.5037 - val_accuracy: 0.8610
    Epoch 64/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0564 - accuracy: 0.9780 - val_loss: 1.0875 - val_accuracy: 0.7273
    Epoch 65/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0601 - accuracy: 0.9776 - val_loss: 0.6925 - val_accuracy: 0.7914
    Epoch 66/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0549 - accuracy: 0.9785 - val_loss: 1.1024 - val_accuracy: 0.7059
    Epoch 67/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0535 - accuracy: 0.9806 - val_loss: 0.9329 - val_accuracy: 0.7219
    Epoch 68/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0555 - accuracy: 0.9785 - val_loss: 0.5477 - val_accuracy: 0.8503
    Epoch 69/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0549 - accuracy: 0.9789 - val_loss: 1.1979 - val_accuracy: 0.7059
    Epoch 70/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0546 - accuracy: 0.9795 - val_loss: 0.3710 - val_accuracy: 0.8877
    Epoch 71/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0535 - accuracy: 0.9791 - val_loss: 0.6753 - val_accuracy: 0.8128
    Epoch 72/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0510 - accuracy: 0.9801 - val_loss: 0.7752 - val_accuracy: 0.7807
    Epoch 73/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0528 - accuracy: 0.9785 - val_loss: 0.9951 - val_accuracy: 0.7380
    Epoch 74/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0499 - accuracy: 0.9814 - val_loss: 0.9847 - val_accuracy: 0.7540
    Epoch 75/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0506 - accuracy: 0.9803 - val_loss: 0.7040 - val_accuracy: 0.7914
    Epoch 76/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0452 - accuracy: 0.9833 - val_loss: 0.7437 - val_accuracy: 0.7861
    Epoch 77/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0494 - accuracy: 0.9808 - val_loss: 0.9447 - val_accuracy: 0.7540
    Epoch 78/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0502 - accuracy: 0.9803 - val_loss: 0.6604 - val_accuracy: 0.8182
    Epoch 79/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0480 - accuracy: 0.9814 - val_loss: 0.6579 - val_accuracy: 0.8289
    Epoch 80/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0456 - accuracy: 0.9831 - val_loss: 0.6554 - val_accuracy: 0.8075
    Epoch 81/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0458 - accuracy: 0.9829 - val_loss: 0.9803 - val_accuracy: 0.7487
    Epoch 82/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0442 - accuracy: 0.9849 - val_loss: 0.6368 - val_accuracy: 0.8289
    Epoch 83/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0425 - accuracy: 0.9839 - val_loss: 0.9355 - val_accuracy: 0.7594
    Epoch 84/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0402 - accuracy: 0.9872 - val_loss: 0.7137 - val_accuracy: 0.8075
    Epoch 85/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0433 - accuracy: 0.9841 - val_loss: 0.8757 - val_accuracy: 0.7701
    Epoch 86/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0416 - accuracy: 0.9847 - val_loss: 0.8325 - val_accuracy: 0.7807
    Epoch 87/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0400 - accuracy: 0.9854 - val_loss: 1.0463 - val_accuracy: 0.7540
    Epoch 88/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0416 - accuracy: 0.9839 - val_loss: 0.6424 - val_accuracy: 0.8128
    Epoch 89/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0405 - accuracy: 0.9847 - val_loss: 0.8652 - val_accuracy: 0.7754
    Epoch 90/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0411 - accuracy: 0.9849 - val_loss: 0.8548 - val_accuracy: 0.7701
    Epoch 91/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0382 - accuracy: 0.9856 - val_loss: 0.6673 - val_accuracy: 0.8396
    Epoch 92/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0417 - accuracy: 0.9831 - val_loss: 0.5029 - val_accuracy: 0.8556
    Epoch 93/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0413 - accuracy: 0.9850 - val_loss: 0.9789 - val_accuracy: 0.7647
    Epoch 94/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0352 - accuracy: 0.9873 - val_loss: 0.8605 - val_accuracy: 0.7701
    Epoch 95/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0380 - accuracy: 0.9854 - val_loss: 0.7986 - val_accuracy: 0.7807
    Epoch 96/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0360 - accuracy: 0.9858 - val_loss: 1.1914 - val_accuracy: 0.7219
    Epoch 97/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0474 - accuracy: 0.9827 - val_loss: 0.7654 - val_accuracy: 0.7701
    Epoch 98/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0378 - accuracy: 0.9860 - val_loss: 0.9057 - val_accuracy: 0.7647
    Epoch 99/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0326 - accuracy: 0.9885 - val_loss: 0.6311 - val_accuracy: 0.8289
    Epoch 100/100
    163/163 [==============================] - 2s 11ms/step - loss: 0.0341 - accuracy: 0.9875 - val_loss: 1.2658 - val_accuracy: 0.7219
    


```python
dfBaseline = pd.DataFrame().from_dict(baselineHistory.history)
dfBaseline
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
      <th>accuracy</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.576326</td>
      <td>0.708781</td>
      <td>0.677273</td>
      <td>0.582888</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.365830</td>
      <td>0.845667</td>
      <td>0.368742</td>
      <td>0.802139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.288074</td>
      <td>0.881135</td>
      <td>0.273962</td>
      <td>0.866310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.228575</td>
      <td>0.902416</td>
      <td>0.230886</td>
      <td>0.882353</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.211185</td>
      <td>0.915644</td>
      <td>0.226919</td>
      <td>0.919786</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.035975</td>
      <td>0.985813</td>
      <td>1.191390</td>
      <td>0.721925</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.047436</td>
      <td>0.982745</td>
      <td>0.765383</td>
      <td>0.770053</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.037772</td>
      <td>0.986005</td>
      <td>0.905660</td>
      <td>0.764706</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.032627</td>
      <td>0.988497</td>
      <td>0.631139</td>
      <td>0.828877</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.034078</td>
      <td>0.987538</td>
      <td>1.265848</td>
      <td>0.721925</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>




```python
fig, ((ax1, ax2)) = plt.subplots(nrows = 1,ncols = 2, figsize = (14,7))
dfBaseline.plot(y = ['loss', 'val_loss'],ax = ax1, title = 'Loss Metrics');
dfBaseline.plot(y = ['accuracy', 'val_accuracy'],ax = ax2, title = 'Accuracy Metrics');
```


![png](output_22_0.png)


### Load Pre-Saved Baseline Model


```python
# baselineModel = keras.models.load_model('./baseline_model/')
```

# Second Model: Advanced Connvulution Model
## Changes From Baseline
The simple dense layer model used as the baseline wasn't taylored to image classification. There are, however, several other layer types that are much more adept at picking up on the data structure of images. The second model will rely on convulutional and pooling layers. 

A true deep dive into the underlying workings of these layers is beyond the scope of this notebook but we will sum them up for convienence.
### Backgroud Information
#### Convulutional layers 
Convulutional layers are the life force of modern neural network image processing. Unlike dense layers which connect every node of a layer to every node of the next layer, convulutional layers only connect small portions of the input layers. This approach is modeled off the human ocular system and is a state of the art way for understanding high dimensional imformation that boils down to a smaller subset of features. Images are the connonical example of this, but there are also several other types of data that can be analyzed effectively in this way. Ultimately, the convulution layers serve as feature extractors. In almost all examples they are combined with traditiaonl dense layers which take the information filtered by the convulutional layers and create prediction based on the normal preceptron model.
#### Pooling Layers
Pooling layers are the balance to convulution layers. While not strictly necessary, pooling layers are often used to decrease the dimensionality of the data. Because dense are more or less required to lie on top of the convulution layers to preform the "thinking" part of the model, we run into a dimensionality problem when transitioning the data from the convulutional layers to the dense layers. Dense layers require 1-dimensional input where as convulutional layers are capable of analyzing and outputing multi-dimensional data (2-dimensions in this case). In order to create this transition we must flatten the 2-dimensional data into 1 dimension which explodes the number of required node connections for the first dense layer. A 256 X 256 pixel image to transition to a dense layer of 20 nodes requires 128,000 connections per filter used in the preceeding convulutional layer. This will quickly get out of hand. The solution is pooling layers. Pooling layers use summary statistics to combine areas of the convulutional layers whereby effectively reducing the output data's dimensionality while preserving most of the information from the inputs. While, reducing the number of features likely means a reduced accuracy rate, the output will still retain enough features to catagorize images with incredible accuracy (this shouldn't be too surprising if you think about how quickly you can recognize objects from your periferal vision.)
## Building the second model
If you want to see the code involved with creating and fitting the convulutional model please open the code below. We will cover the methodology in narrative here. Because we want to take advantage of the Imagedatagenterator class' ability to augment data the first thing we are going to do is create new generator objects. The training generator will include a number of augmentation parameters that were determined through trial and error. 
* A short comment on the parameters: Augmentation was extremely beneficial to the convulutional models preformace. The actual numbers used reflect a moderate amount of augmentation. Ultimately, no reflections or rotations were utilized as they appeared to have a strictly negative impact on model preformace. One possible reason for this couls be the standard directionality of the x-rays themselves which were all oriented in the same manner. Ultimately, if there's no standardization in the image uploading process then rotational and reflection augmentations may have a more positive impact. But, with this specific dataset, they were not considered. 

There are some specific data manipulations that are implemented in order to better utalize colab resources and avoid having to access the directory system for each batch. If you wish to inspect the cod used to create the second model you can expand the following cells and go wild.

### Model Layers
The second model is composed of the following architecture. 
* [Insert layers list for convulutional network]
* [Insert layers for dense network]

Many of these decisions were based on some initial experimentation and third party research. The specific design was based on designs featured from two papers. One paper is titled "FRD-CNN: Object detection based on small-scale convolutional neural networks and feature reuse" and "Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization" both of which are articles featured in the journal "scientific reports"

The general idea for the network architecture is to have two convulution layers followed by a pooling a layer. This pattern is repeated between 5 to 10 times and then the resulting filters are funneled into the dense layers and finally into the binary decision layer. The two literature resources have additional features that are specifically designed for image segmentation. I left these components out of my architecture. In addition, I added additional batch normalization layers in order to further decrease the probability of overfitting. 

The funnel chose for connecting my convulutional and dense layers was a global average pooling layer. This is a specific type of pooling layer than creates a single value from the each of the filters being passed to it. The general idea of a convulutional layer is to extract features from the initial images. The global average pooling layer then represents each of these features with a single values that hopefully the dense layers can use to learn. The reason for using this particular type of pooling layer is to avoid overfitting in the dense layers. Images have so much information from the number of pixels that dense layers have a tendency to memorize the training images. This is a phenomenon in which the neural network will have incredible accuracy on the training images but struggle to generalize to a testing set. 

For a full list of model parameters please see the code sections. In general, relu activation was used for the convulutional layers and orthogonol kernals were used to initialize the kernal state.  

### Model Fitting
<img src = 'second-model-training-metrics.png'>

## Second Model Evaluation
We begin with a visual of the confusion matrix, often a staple summary of a model's preformance. 
<img src = 'second-model-confustion-matrix.png'>

### General Preformace

The second model has performed incredibly well in comparison to the baseline. The training curves jitter as the model attempts to find the underlying features of the data and becomes incredibly smooth as it settles on a final weight configuration. The right end of the curve suggest the model isn't massivly overfitting or failing to converge which is a good position to be in for increased generalization. 

### Accuracy

Our model has landed on 93% overall accuracy during the best training iteration, a massive improvement from the baseline and overall an impressive achievement given the limited number of images in the dataset. 

### Pneumonia Recall

The recall on the positive case was 97%. This is almost optimal given the task is medical diagnosis and catching the positive case is more important than strict accuracy. 

### Non-Pneumonia Recall

Recall preformace on the negative class was 87%. Lower than the positive case but if the choice was to increase the negative case accuracy at the expense of the positive case accuracy then the model would be less effective overall even if there was a slight increase in accuracy overall. 

## Code To Create The Second Model


```python
# We don't do any image rotations or reflections because they tended to yield worse results
# This isn't likely to reduce the model ability to generalize because images should be uploaded
# in a standard way
trainGenerator = ImageDataGenerator(
    width_shift_range=0.3,
    height_shift_range=0.3, 
    zoom_range=0.15,
    )

# no augments on the test generator
testGenerator = ImageDataGenerator()

```


```python
# The flowed data from the top of the notebook. This will be fed directly into the neural networks
 
train = trainGenerator.flow(
    trainImages,
    y = trainImageLabels, 
    shuffle = False,
    batch_size = 64,
)
# The split off validation set. This is fed directly into the neural networks
val = testGenerator.flow(
    valImages,
    y = valImageLabels, 
    shuffle = False,
    batch_size = 64,
 
)
 
# The flowed data from the top of the notebook. This will be fed directly into the neural networks
test = testGenerator.flow(
    testImages,
    y = testImageLabels, 
    shuffle = False,
    batch_size = 64,
 
)
```


```python
def create_model_visuals(model, trainGenerator, valGenerator, testGenerator, batch_size, epochs, class_weight, train = True,kwargs = {}):
  params = locals()
  def confustion_matrix(y, y_hat, normalize = 'true'):
    fig, ax = plt.subplots(1,1,figsize = (7,6))
    matrix = skm.confusion_matrix(y, y_hat, normalize=normalize,)
    sns.heatmap(matrix, cmap = 'Blues', annot=True, ax = ax)
    ax.set(
      title = 'Confustion Matrix',
      xlabel = 'Predicted Label',
      ylabel = 'True Label'
    )
  if train == True:
    modelHistory = model.fit(
        trainGenerator,
        batch_size=batch_size,
        epochs=epochs,
        class_weight = class_weight,
        **kwargs)
    model.evaluate(testGenerator)
    dfModel = pd.DataFrame().from_dict(modelHistory.history)
    fig, ((ax1,ax2),(ax3,ax4),(ax5, ax6)) = plt.subplots(nrows = 3,ncols = 2, figsize = (18,7))
    dfModel.plot(y = ['loss', 'val_loss'],ax = ax1, title = 'Loss Metrics', xlabel = 'Training Generation', ylabel = 'Loss score');
    dfModel.plot(y = ['accuracy', 'val_accuracy'],ax = ax2, title = 'Accuracy',xlabel = 'Training Generation', ylabel = 'Accuracy Percentage');
    dfModel.plot(y = ['auc', 'val_auc'],ax = ax3, title = 'Area Under The Curve',xlabel = 'Training Generation', ylabel = 'Area Under The Curve');
    dfModel.plot(y = ['square_hinge', 'val_square_hinge'],ax = ax4, title = 'Square Hinge',xlabel = 'Training Generation', ylabel = 'Square Hinge');
    dfModel.plot(y = ['val_true_positives'],ax = ax5, title = 'Val True Positives',xlabel = 'Training Generation', ylabel = 'Number of True Positives');
    dfModel.plot(y = [ 'val_true_negatives'],ax = ax6, title = 'Val True Negatives',xlabel = 'Training Generation', ylabel = 'Number of True Negatives');
    plt.tight_layout()
    plt.show()
     
  else:
    dfModel = None
 
  y_test_hat = np.where(model.predict(testGenerator) > .5, 1,0).flatten()
  y_test = testGenerator.y
  confustion_matrix(y_test, y_test_hat)
  dfTest = pd.DataFrame.from_dict(skm.classification_report(y_test, y_test_hat, output_dict=True))
  display(dfTest)
 
  return dfModel,params
```


```python
def scale_weight(num): 
  weights = compute_class_weight(
    'balanced',
    classes=[0, 1],
    y = trainImageLabels)
  weights = {0:weights[0]*num, 1:weights[1]/num}
  return weights
```


```python
secondModel = Sequential()
secondModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      input_shape = (256,256,3)
      )
    )
secondModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())
 
secondModel.add(layers.MaxPooling2D(2))
 
secondModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
secondModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())
 
secondModel.add(layers.MaxPooling2D(2))
 
secondModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
secondModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())
 
 
secondModel.add(layers.MaxPooling2D(2))
 
 
secondModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
secondModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())
 
 
secondModel.add(layers.MaxPooling2D(2))
 
secondModel.add(
    layers.Conv2D(
      256,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
secondModel.add(
    layers.Conv2D(
      256,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())
 
secondModel.add(layers.MaxPooling2D(2))
 
secondModel.add(
    layers.Conv2D(
      512,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
secondModel.add(
    layers.Conv2D(
      512,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
secondModel.add(layers.BatchNormalization())

 
secondModel.add(layers.GlobalAveragePooling2D())
 
 
secondModel.add(layers.Dense(
    128, 
    kernel_initializer=keras.initializers.Orthogonal(),
    activity_regularizer = keras.regularizers.L2(.02),
    )
)
 
secondModel.add(layers.LeakyReLU())
 
secondModel.add(layers.BatchNormalization())
 
secondModel.add(layers.Dense(
    6,
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal())
)
 
secondModel.add(layers.BatchNormalization())
 
 
secondModel.add(layers.Dense(
    6, 
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal(),
    ))
 
secondModel.add(layers.Dense(1, activation = 'sigmoid'))
 
 
secondModel.compile(
    optimizer = keras.optimizers.RMSprop(
        learning_rate=.001,
        # momentum = .1
        ),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [
      keras.metrics.BinaryAccuracy(name = 'accuracy'),
      keras.metrics.AUC(name = 'auc'),
      keras.metrics.SquaredHinge(name = 'square_hinge'),
      keras.metrics.TruePositives(name='true_positives'), 
      keras.metrics.TrueNegatives(name = 'true_negatives')]
)
secondModel.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 256, 256, 16)      448       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 256, 256, 16)      2320      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 256, 256, 16)      64        
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 128, 128, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 128, 128, 32)      4640      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 128, 128, 32)      9248      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 128, 128, 32)      128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 64, 64, 32)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 64, 64, 64)        18496     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 64, 64, 64)        36928     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 64, 64, 64)        256       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 32, 32, 128)       73856     
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 32, 32, 128)       147584    
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 32, 32, 128)       512       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 16, 16, 256)       295168    
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 16, 16, 256)       590080    
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 16, 16, 256)       1024      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 8, 8, 256)         0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 8, 8, 512)         1180160   
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 8, 8, 512)         2048      
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 128)               65664     
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 128)               0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 128)               512       
    _________________________________________________________________
    dense_3 (Dense)              (None, 6)                 774       
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 6)                 24        
    _________________________________________________________________
    dense_4 (Dense)              (None, 6)                 42        
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 7         
    =================================================================
    Total params: 4,789,791
    Trainable params: 4,787,507
    Non-trainable params: 2,284
    _________________________________________________________________
    


```python
# we define the filepath where any results will be saved
secondModel_filepath = os.path.join('second_model')

# the Callbacks for keras.Model.fit() method. See Keras documentation for more info.
secondModelEarlyStop = EarlyStopping(patience= 9, mode = 'auto', restore_best_weights=False, monitor='val_loss')
secondModelCheckpoint = ModelCheckpoint(secondModel_filepath,save_best_only=True, monitor='val_loss')
secondModelLRAdjust = ReduceLROnPlateau(monitor = 'val_loss', factor = .5, patience=2, min_delta=.00000000001)

# This cell runs the fitting 
secondModelHistory,params = create_model_visuals(
    model = secondModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 150, 
    batch_size=64,
    class_weight = scale_weight(1.5),
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[secondModelCheckpoint, secondModelEarlyStop, secondModelLRAdjust]})
```

    Epoch 1/150
    82/82 [==============================] - ETA: 0s - loss: 0.5391 - accuracy: 0.7287 - auc: 0.8735 - square_hinge: 0.5638 - true_positives: 2577.0000 - true_negatives: 1224.0000WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    This property should not be used in TensorFlow 2.0, as updates are applied automatically.
    INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 72s 876ms/step - loss: 0.5391 - accuracy: 0.7287 - auc: 0.8735 - square_hinge: 0.5638 - true_positives: 2577.0000 - true_negatives: 1224.0000 - val_loss: 9.1411 - val_accuracy: 0.3743 - val_auc: 0.5000 - val_square_hinge: 0.9425 - val_true_positives: 0.0000e+00 - val_true_negatives: 70.0000
    Epoch 2/150
    82/82 [==============================] - ETA: 0s - loss: 0.3893 - accuracy: 0.7910 - auc: 0.9130 - square_hinge: 0.4934 - true_positives: 2861.0000 - true_negatives: 1265.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 869ms/step - loss: 0.3893 - accuracy: 0.7910 - auc: 0.9130 - square_hinge: 0.4934 - true_positives: 2861.0000 - true_negatives: 1265.0000 - val_loss: 7.3311 - val_accuracy: 0.3743 - val_auc: 0.4716 - val_square_hinge: 0.9744 - val_true_positives: 0.0000e+00 - val_true_negatives: 70.0000
    Epoch 3/150
    82/82 [==============================] - ETA: 0s - loss: 0.3270 - accuracy: 0.8296 - auc: 0.9325 - square_hinge: 0.4541 - true_positives: 3040.0000 - true_negatives: 1287.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 866ms/step - loss: 0.3270 - accuracy: 0.8296 - auc: 0.9325 - square_hinge: 0.4541 - true_positives: 3040.0000 - true_negatives: 1287.0000 - val_loss: 4.0576 - val_accuracy: 0.6257 - val_auc: 0.5000 - val_square_hinge: 1.3943 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 4/150
    82/82 [==============================] - ETA: 0s - loss: 0.3200 - accuracy: 0.8395 - auc: 0.9330 - square_hinge: 0.4433 - true_positives: 3095.0000 - true_negatives: 1284.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 866ms/step - loss: 0.3200 - accuracy: 0.8395 - auc: 0.9330 - square_hinge: 0.4433 - true_positives: 3095.0000 - true_negatives: 1284.0000 - val_loss: 1.8318 - val_accuracy: 0.6257 - val_auc: 0.5000 - val_square_hinge: 1.4061 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 5/150
    82/82 [==============================] - ETA: 0s - loss: 0.2821 - accuracy: 0.8503 - auc: 0.9434 - square_hinge: 0.4265 - true_positives: 3143.0000 - true_negatives: 1292.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 864ms/step - loss: 0.2821 - accuracy: 0.8503 - auc: 0.9434 - square_hinge: 0.4265 - true_positives: 3143.0000 - true_negatives: 1292.0000 - val_loss: 1.3321 - val_accuracy: 0.6257 - val_auc: 0.5357 - val_square_hinge: 1.4195 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 6/150
    82/82 [==============================] - ETA: 0s - loss: 0.2735 - accuracy: 0.8660 - auc: 0.9498 - square_hinge: 0.4134 - true_positives: 3228.0000 - true_negatives: 1289.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 72s 877ms/step - loss: 0.2735 - accuracy: 0.8660 - auc: 0.9498 - square_hinge: 0.4134 - true_positives: 3228.0000 - true_negatives: 1289.0000 - val_loss: 1.0960 - val_accuracy: 0.6257 - val_auc: 0.9240 - val_square_hinge: 1.3097 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 7/150
    82/82 [==============================] - ETA: 0s - loss: 0.2570 - accuracy: 0.8614 - auc: 0.9570 - square_hinge: 0.4083 - true_positives: 3196.0000 - true_negatives: 1297.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 863ms/step - loss: 0.2570 - accuracy: 0.8614 - auc: 0.9570 - square_hinge: 0.4083 - true_positives: 3196.0000 - true_negatives: 1297.0000 - val_loss: 0.9495 - val_accuracy: 0.7540 - val_auc: 0.8708 - val_square_hinge: 1.0587 - val_true_positives: 117.0000 - val_true_negatives: 24.0000
    Epoch 8/150
    82/82 [==============================] - 67s 812ms/step - loss: 0.2353 - accuracy: 0.8823 - auc: 0.9622 - square_hinge: 0.3927 - true_positives: 3301.0000 - true_negatives: 1301.0000 - val_loss: 21.6429 - val_accuracy: 0.6257 - val_auc: 0.9278 - val_square_hinge: 1.2933 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 9/150
    82/82 [==============================] - 66s 810ms/step - loss: 0.2242 - accuracy: 0.8884 - auc: 0.9667 - square_hinge: 0.3838 - true_positives: 3330.0000 - true_negatives: 1304.0000 - val_loss: 1.3247 - val_accuracy: 0.6257 - val_auc: 0.6571 - val_square_hinge: 1.4372 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 10/150
    82/82 [==============================] - ETA: 0s - loss: 0.1867 - accuracy: 0.9087 - auc: 0.9759 - square_hinge: 0.3619 - true_positives: 3432.0000 - true_negatives: 1308.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 69s 846ms/step - loss: 0.1867 - accuracy: 0.9087 - auc: 0.9759 - square_hinge: 0.3619 - true_positives: 3432.0000 - true_negatives: 1308.0000 - val_loss: 0.9139 - val_accuracy: 0.6364 - val_auc: 0.9526 - val_square_hinge: 1.3193 - val_true_positives: 117.0000 - val_true_negatives: 2.0000
    Epoch 11/150
    82/82 [==============================] - ETA: 0s - loss: 0.1680 - accuracy: 0.9195 - auc: 0.9810 - square_hinge: 0.3519 - true_positives: 3485.0000 - true_negatives: 1311.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 70s 859ms/step - loss: 0.1680 - accuracy: 0.9195 - auc: 0.9810 - square_hinge: 0.3519 - true_positives: 3485.0000 - true_negatives: 1311.0000 - val_loss: 0.3812 - val_accuracy: 0.8717 - val_auc: 0.9416 - val_square_hinge: 0.7116 - val_true_positives: 110.0000 - val_true_negatives: 53.0000
    Epoch 12/150
    82/82 [==============================] - ETA: 0s - loss: 0.1572 - accuracy: 0.9294 - auc: 0.9815 - square_hinge: 0.3421 - true_positives: 3541.0000 - true_negatives: 1307.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 868ms/step - loss: 0.1572 - accuracy: 0.9294 - auc: 0.9815 - square_hinge: 0.3421 - true_positives: 3541.0000 - true_negatives: 1307.0000 - val_loss: 0.2376 - val_accuracy: 0.8984 - val_auc: 0.9930 - val_square_hinge: 0.6496 - val_true_positives: 117.0000 - val_true_negatives: 51.0000
    Epoch 13/150
    82/82 [==============================] - 66s 811ms/step - loss: 0.1564 - accuracy: 0.9310 - auc: 0.9796 - square_hinge: 0.3412 - true_positives: 3545.0000 - true_negatives: 1311.0000 - val_loss: 0.3046 - val_accuracy: 0.8770 - val_auc: 0.9853 - val_square_hinge: 0.7592 - val_true_positives: 116.0000 - val_true_negatives: 48.0000
    Epoch 14/150
    82/82 [==============================] - ETA: 0s - loss: 0.1524 - accuracy: 0.9264 - auc: 0.9842 - square_hinge: 0.3410 - true_positives: 3523.0000 - true_negatives: 1309.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 71s 864ms/step - loss: 0.1524 - accuracy: 0.9264 - auc: 0.9842 - square_hinge: 0.3410 - true_positives: 3523.0000 - true_negatives: 1309.0000 - val_loss: 0.2086 - val_accuracy: 0.9198 - val_auc: 0.9910 - val_square_hinge: 0.4743 - val_true_positives: 104.0000 - val_true_negatives: 68.0000
    Epoch 15/150
    82/82 [==============================] - 67s 821ms/step - loss: 0.1353 - accuracy: 0.9413 - auc: 0.9849 - square_hinge: 0.3284 - true_positives: 3596.0000 - true_negatives: 1314.0000 - val_loss: 0.7860 - val_accuracy: 0.7807 - val_auc: 0.9211 - val_square_hinge: 0.5836 - val_true_positives: 76.0000 - val_true_negatives: 70.0000
    Epoch 16/150
    82/82 [==============================] - 68s 828ms/step - loss: 0.1409 - accuracy: 0.9402 - auc: 0.9847 - square_hinge: 0.3316 - true_positives: 3591.0000 - true_negatives: 1313.0000 - val_loss: 0.3451 - val_accuracy: 0.8610 - val_auc: 0.9872 - val_square_hinge: 0.5233 - val_true_positives: 91.0000 - val_true_negatives: 70.0000
    Epoch 17/150
    82/82 [==============================] - 68s 828ms/step - loss: 0.1203 - accuracy: 0.9500 - auc: 0.9890 - square_hinge: 0.3218 - true_positives: 3641.0000 - true_negatives: 1314.0000 - val_loss: 0.4352 - val_accuracy: 0.8128 - val_auc: 0.9789 - val_square_hinge: 0.8824 - val_true_positives: 117.0000 - val_true_negatives: 35.0000
    Epoch 18/150
    82/82 [==============================] - 68s 827ms/step - loss: 0.1152 - accuracy: 0.9513 - auc: 0.9906 - square_hinge: 0.3185 - true_positives: 3643.0000 - true_negatives: 1319.0000 - val_loss: 0.7904 - val_accuracy: 0.6845 - val_auc: 0.9684 - val_square_hinge: 0.6156 - val_true_positives: 58.0000 - val_true_negatives: 70.0000
    Epoch 19/150
    82/82 [==============================] - 69s 839ms/step - loss: 0.1043 - accuracy: 0.9582 - auc: 0.9924 - square_hinge: 0.3117 - true_positives: 3681.0000 - true_negatives: 1317.0000 - val_loss: 0.3704 - val_accuracy: 0.8235 - val_auc: 0.9996 - val_square_hinge: 0.8305 - val_true_positives: 117.0000 - val_true_negatives: 37.0000
    Epoch 20/150
    82/82 [==============================] - 68s 834ms/step - loss: 0.1001 - accuracy: 0.9586 - auc: 0.9937 - square_hinge: 0.3096 - true_positives: 3686.0000 - true_negatives: 1314.0000 - val_loss: 0.8926 - val_accuracy: 0.6684 - val_auc: 0.9916 - val_square_hinge: 1.2557 - val_true_positives: 117.0000 - val_true_negatives: 8.0000
    Epoch 21/150
    82/82 [==============================] - ETA: 0s - loss: 0.0865 - accuracy: 0.9640 - auc: 0.9937 - square_hinge: 0.3030 - true_positives: 3705.0000 - true_negatives: 1323.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 72s 877ms/step - loss: 0.0865 - accuracy: 0.9640 - auc: 0.9937 - square_hinge: 0.3030 - true_positives: 3705.0000 - true_negatives: 1323.0000 - val_loss: 0.1369 - val_accuracy: 0.9412 - val_auc: 0.9998 - val_square_hinge: 0.5533 - val_true_positives: 117.0000 - val_true_negatives: 59.0000
    Epoch 22/150
    82/82 [==============================] - 69s 838ms/step - loss: 0.0915 - accuracy: 0.9636 - auc: 0.9943 - square_hinge: 0.3047 - true_positives: 3710.0000 - true_negatives: 1316.0000 - val_loss: 0.3228 - val_accuracy: 0.8556 - val_auc: 0.9986 - val_square_hinge: 0.7835 - val_true_positives: 117.0000 - val_true_negatives: 43.0000
    Epoch 23/150
    82/82 [==============================] - 69s 840ms/step - loss: 0.0934 - accuracy: 0.9628 - auc: 0.9935 - square_hinge: 0.3057 - true_positives: 3703.0000 - true_negatives: 1319.0000 - val_loss: 0.4573 - val_accuracy: 0.7861 - val_auc: 0.9996 - val_square_hinge: 0.9138 - val_true_positives: 117.0000 - val_true_negatives: 30.0000
    Epoch 24/150
    82/82 [==============================] - ETA: 0s - loss: 0.0818 - accuracy: 0.9668 - auc: 0.9956 - square_hinge: 0.3002 - true_positives: 3721.0000 - true_negatives: 1322.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 73s 890ms/step - loss: 0.0818 - accuracy: 0.9668 - auc: 0.9956 - square_hinge: 0.3002 - true_positives: 3721.0000 - true_negatives: 1322.0000 - val_loss: 0.0856 - val_accuracy: 0.9679 - val_auc: 0.9989 - val_square_hinge: 0.4122 - val_true_positives: 111.0000 - val_true_negatives: 70.0000
    Epoch 25/150
    82/82 [==============================] - 68s 834ms/step - loss: 0.0785 - accuracy: 0.9688 - auc: 0.9956 - square_hinge: 0.2982 - true_positives: 3728.0000 - true_negatives: 1325.0000 - val_loss: 0.1269 - val_accuracy: 0.9626 - val_auc: 0.9993 - val_square_hinge: 0.4165 - val_true_positives: 110.0000 - val_true_negatives: 70.0000
    Epoch 26/150
    82/82 [==============================] - ETA: 0s - loss: 0.0816 - accuracy: 0.9674 - auc: 0.9953 - square_hinge: 0.3005 - true_positives: 3723.0000 - true_negatives: 1323.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 73s 887ms/step - loss: 0.0816 - accuracy: 0.9674 - auc: 0.9953 - square_hinge: 0.3005 - true_positives: 3723.0000 - true_negatives: 1323.0000 - val_loss: 0.0670 - val_accuracy: 0.9733 - val_auc: 0.9998 - val_square_hinge: 0.4061 - val_true_positives: 112.0000 - val_true_negatives: 70.0000
    Epoch 27/150
    82/82 [==============================] - ETA: 0s - loss: 0.0834 - accuracy: 0.9657 - auc: 0.9949 - square_hinge: 0.3011 - true_positives: 3716.0000 - true_negatives: 1321.0000INFO:tensorflow:Assets written to: second_model/assets
    82/82 [==============================] - 72s 880ms/step - loss: 0.0834 - accuracy: 0.9657 - auc: 0.9949 - square_hinge: 0.3011 - true_positives: 3716.0000 - true_negatives: 1321.0000 - val_loss: 0.0523 - val_accuracy: 0.9840 - val_auc: 1.0000 - val_square_hinge: 0.4041 - val_true_positives: 114.0000 - val_true_negatives: 70.0000
    Epoch 28/150
    82/82 [==============================] - 67s 814ms/step - loss: 0.0784 - accuracy: 0.9688 - auc: 0.9947 - square_hinge: 0.2988 - true_positives: 3727.0000 - true_negatives: 1326.0000 - val_loss: 0.0621 - val_accuracy: 0.9733 - val_auc: 0.9994 - val_square_hinge: 0.4085 - val_true_positives: 113.0000 - val_true_negatives: 69.0000
    Epoch 29/150
    82/82 [==============================] - 66s 810ms/step - loss: 0.0846 - accuracy: 0.9666 - auc: 0.9950 - square_hinge: 0.3012 - true_positives: 3723.0000 - true_negatives: 1319.0000 - val_loss: 0.1783 - val_accuracy: 0.9519 - val_auc: 0.9986 - val_square_hinge: 0.4333 - val_true_positives: 108.0000 - val_true_negatives: 70.0000
    Epoch 30/150
    82/82 [==============================] - 67s 818ms/step - loss: 0.0702 - accuracy: 0.9745 - auc: 0.9962 - square_hinge: 0.2930 - true_positives: 3757.0000 - true_negatives: 1326.0000 - val_loss: 0.0713 - val_accuracy: 0.9733 - val_auc: 0.9994 - val_square_hinge: 0.4114 - val_true_positives: 113.0000 - val_true_negatives: 69.0000
    Epoch 31/150
    82/82 [==============================] - 67s 814ms/step - loss: 0.0818 - accuracy: 0.9695 - auc: 0.9946 - square_hinge: 0.2992 - true_positives: 3736.0000 - true_negatives: 1321.0000 - val_loss: 0.0737 - val_accuracy: 0.9733 - val_auc: 0.9995 - val_square_hinge: 0.4073 - val_true_positives: 112.0000 - val_true_negatives: 70.0000
    Epoch 32/150
    82/82 [==============================] - 67s 818ms/step - loss: 0.0800 - accuracy: 0.9712 - auc: 0.9941 - square_hinge: 0.2975 - true_positives: 3743.0000 - true_negatives: 1323.0000 - val_loss: 0.0703 - val_accuracy: 0.9679 - val_auc: 0.9994 - val_square_hinge: 0.4088 - val_true_positives: 112.0000 - val_true_negatives: 69.0000
    Epoch 33/150
    82/82 [==============================] - 66s 810ms/step - loss: 0.0763 - accuracy: 0.9697 - auc: 0.9946 - square_hinge: 0.2968 - true_positives: 3731.0000 - true_negatives: 1327.0000 - val_loss: 0.0588 - val_accuracy: 0.9733 - val_auc: 0.9994 - val_square_hinge: 0.4076 - val_true_positives: 113.0000 - val_true_negatives: 69.0000
    Epoch 34/150
    82/82 [==============================] - 65s 795ms/step - loss: 0.0771 - accuracy: 0.9686 - auc: 0.9949 - square_hinge: 0.2974 - true_positives: 3730.0000 - true_negatives: 1322.0000 - val_loss: 0.0740 - val_accuracy: 0.9733 - val_auc: 0.9995 - val_square_hinge: 0.4080 - val_true_positives: 112.0000 - val_true_negatives: 70.0000
    Epoch 35/150
    82/82 [==============================] - 65s 795ms/step - loss: 0.0718 - accuracy: 0.9699 - auc: 0.9956 - square_hinge: 0.2955 - true_positives: 3730.0000 - true_negatives: 1329.0000 - val_loss: 0.0809 - val_accuracy: 0.9733 - val_auc: 0.9995 - val_square_hinge: 0.4090 - val_true_positives: 112.0000 - val_true_negatives: 70.0000
    Epoch 36/150
    82/82 [==============================] - 66s 802ms/step - loss: 0.0804 - accuracy: 0.9711 - auc: 0.9942 - square_hinge: 0.2978 - true_positives: 3740.0000 - true_negatives: 1325.0000 - val_loss: 0.0727 - val_accuracy: 0.9733 - val_auc: 0.9994 - val_square_hinge: 0.4089 - val_true_positives: 112.0000 - val_true_negatives: 70.0000
    7/7 [==============================] - 0s 65ms/step - loss: 0.3174 - accuracy: 0.8902 - auc: 0.9497 - square_hinge: 0.6273 - true_positives: 258.0000 - true_negatives: 131.0000
    


![png](output_33_1.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.897260</td>
      <td>0.886598</td>
      <td>0.89016</td>
      <td>0.891929</td>
      <td>0.890599</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.798780</td>
      <td>0.945055</td>
      <td>0.89016</td>
      <td>0.871918</td>
      <td>0.890160</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.845161</td>
      <td>0.914894</td>
      <td>0.89016</td>
      <td>0.880027</td>
      <td>0.888724</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.89016</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_33_3.png)



```python
# save the training information to a dataframe for future visualization.
secondModelHistory.to_csv('secondModel-train-info.csv')
```


```python
# The previous fitting method saves the model with the least validation loss score, but also prints out
# the model when the early stopping callback stops the fitting. Because of this we get two models from the previous cell

# save the model spit out by the fit method
keras.models.save_model(secondModel, 'second_model_non_checkpoint/')
#load the model saved by the checkpoint callback
secondModel = keras.models.load_model('second_model/')
#show the eval metrics for the check pointed model
keras.models.save_model(secondModel, 'second_model_checkpoint/')

secondModel = keras.models.load_model('second_model_checkpoint/')
secondModelHistory,params = create_model_visuals(
    model = secondModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 100, 
    batch_size=64,
    class_weight = scale_weight(1),
    train = False,
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[secondModelCheckpoint, secondModelEarlyStop]})
```

    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/second_model_non_checkpoint/assets
    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/second_model_checkpoint/assets
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.908397</td>
      <td>0.852941</td>
      <td>0.869565</td>
      <td>0.880669</td>
      <td>0.873753</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.725610</td>
      <td>0.956044</td>
      <td>0.869565</td>
      <td>0.840827</td>
      <td>0.869565</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.806780</td>
      <td>0.901554</td>
      <td>0.869565</td>
      <td>0.854167</td>
      <td>0.865987</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.869565</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_35_2.png)


# Third Model: Undersampling The Majority Class
## Changes From The Second Model
In building the second model, class imbalance was a key issue. The first step in refining the model is to determine to what extent this imbalance is affecting the overal performace. The third model is comprised of the same architecture as the second model. The input data is an undersampled version of the original, such that we've randomly selected 1341 pnuemonia images to match the 1341 normal images. In addition, the following model has slightly different callbacks and class weights to compensate for this shift in class proportions.

## Building Third Model

### Data Preperation
### Model Architecture
### Model Fitting

<img src = 'third-model-training-metrics.png'>

## Third Model Evaluation
### General Performace
The third model had the following confusion matrix

<img src = 'third-model-confusion-matrix.png'>

The model preformed substantially better than the baseline case. The training curves for this model are more dubious than the second model's. The first ten or so epochs roughly mirror the learning from the preievous case but the validation set loss then rapidly decreases below the training set loss. This continues until the end of the training. Oddly, the training set accuracy only managed to obtain 82% accuracy before the earlystopping callback ended the '.fit' method. This is proof that the model wasn't simply memorizing the training images - likely thanks to the data augmentation. What is more likely here is that reducing the dataset so significantly by undersampling may have limited the total amount of underanding the model could glean from the data. The suprieror validation metics lend credence to this theory as the images in the validation set were not augmented and easier to predict. One thing of note for the third model is the close proximity between the testing and training metrics. One can infer the reason is due to the learned patterns in the training set being generalizable.  

In general the model could still find use in classifying the problem at hand given that it is significantly better than the baseline. However, the second model is proof that much more capable models exist and should be utilized over this one. 

### Accuracy
The model's overall accuracy was 81%.

### Pneumonia Recall

The recall on the positive case was 90%.  

### Non-Pneumonia Recall

The recall on the negative case was 66%.  

### Comparison: Third Model vs Second Model

Not much brain power needs to be put into this comparison. The original motivation for the third model was to see if the class imbalace in the data was causing decreased accuracy. It seems that by undersampling, whereby decreasing the total number of images the model saw during training, there was a substantial drop in performace. This is likely the result of decreasing the total information the model was able to learn on. Specifically, given the observed numbers indicate the model was essentially the same when classifying the negative class, the extra images in the positive class that were discarded likely aided in classifying the positive case, and didn't inhibit identifying the negative case. This speaks to the nature of the classification problem, such that more data will likely leed to better accuracy and precision for both the positive and negative cases. 

## Code To Create The Third Model


```python
# This cell preforms the undersampling

# We begin by sorting the training data by class
x_p = trainImages[trainImageLabels == 1]
x_n = trainImages[trainImageLabels == 0]

# we get the total length of the minority class. This will determine how many majority class data points we pull
num = x_n.shape[0]

# make the number of data points into a nparray. 
index_list = np.arange(0,num)

# Take a random subsample of the same number of the minority class
sliceIndecies = np.random.choice(index_list, (num,), replace = False)
x_p_Resampled = trainImages[sliceIndecies]
y_p_resampled = np.array([1 for i in range(num)])
y_n_resampled = np.array([0 for i in range(num)])

# combine the resampled images with their proper labels
train_images_resampled = np.vstack((x_p_Resampled,x_n))
train_labels_resampled = np.hstack((y_p_resampled,y_n_resampled))

# shuffle the images around so the model doesn't pick up on the label ordering. 
index = np.random.permutation(train_images_resampled.shape[0])
train_images_resampled, train_labels_resampled = train_images_resampled[index], train_labels_resampled[index]
```


```python
# Create image generators and fit to the new under sampled training data

trainGenerator = ImageDataGenerator(
    rotation_range=45, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    )
trainGenerator.fit(train_images_resampled)

# we re-run the train generator for consisstancy
testGenerator = ImageDataGenerator(
)
testGenerator.fit(testImages)
```


```python
# Again we use the .flow method to use all our images at once rather than waiting for the steam from directory
 
train = trainGenerator.flow(
    train_images_resampled,
    y = train_labels_resampled, 
    shuffle = False,
    batch_size = 64,)

# The split off validation set. This is fed directly into the neural networks
val = testGenerator.flow(
    valImages,
    y = valImageLabels, 
    shuffle = False,
    batch_size = 64,)
 
# Again we use the .flow method to use all our images at once rather than waiting for the steam from directory
test = testGenerator.flow(
    testImages,
    y = testImageLabels, 
    shuffle = False,
    batch_size = 64,)
```


```python
thirdModel = Sequential()
thirdModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      input_shape = (256,256,3)
      )
    )
thirdModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
thirdModel.add(layers.BatchNormalization())
 
thirdModel.add(layers.MaxPooling2D(2))
 
thirdModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
thirdModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
thirdModel.add(layers.BatchNormalization())
 
thirdModel.add(layers.MaxPooling2D(2))
 
thirdModel.add(
    layers.Conv2D(
      48,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
thirdModel.add(
    layers.Conv2D(
      48,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
thirdModel.add(layers.BatchNormalization())
 
 
thirdModel.add(layers.MaxPooling2D(2))
 

thirdModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
thirdModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
thirdModel.add(layers.BatchNormalization())
 
 
thirdModel.add(layers.MaxPooling2D(2))

thirdModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
thirdModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )

thirdModel.add(layers.BatchNormalization())


thirdModel.add(layers.GlobalAveragePooling2D())
 
 
thirdModel.add(layers.Dense(
    128, 
    kernel_initializer=keras.initializers.Orthogonal(),
    # activation = 'relu'
    # activity_regularizer = keras.regularizers.L2(.05),
    )
)

# thirdModel.add(layers.LeakyReLU())
 
thirdModel.add(layers.Dropout(.2))
 
thirdModel.add(layers.Dense(
    16,
    # activity_regularizer = keras.regularizers.L2(.02),
    activation = 'tanh',
    kernel_initializer = keras.initializers.Orthogonal())
)
 
# thirdModel.add(layers.LeakyReLU())

 
# thirdModel.add(layers.Dense(
#     4, 
#     activation= 'tanh',
#     kernel_initializer = keras.initializers.Orthogonal(),
#     ))
 
thirdModel.add(layers.Dense(1, activation = 'sigmoid'))
 
 
thirdModel.compile(
    optimizer = keras.optimizers.RMSprop(
        learning_rate=.001,
        rho = .8,),
    loss = keras.losses.BinaryCrossentropy(),
    # loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=.26, gamma=1.1),
    metrics = [keras.metrics.BinaryAccuracy(name = 'accuracy'),keras.metrics.TruePositives(name='true_positives'), keras.metrics.TrueNegatives(name = 'true_negatives')]
)
thirdModel.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_70 (Conv2D)           (None, 256, 256, 16)      448       
    _________________________________________________________________
    conv2d_71 (Conv2D)           (None, 256, 256, 16)      2320      
    _________________________________________________________________
    batch_normalization_35 (Batc (None, 256, 256, 16)      64        
    _________________________________________________________________
    max_pooling2d_28 (MaxPooling (None, 128, 128, 16)      0         
    _________________________________________________________________
    conv2d_72 (Conv2D)           (None, 128, 128, 32)      4640      
    _________________________________________________________________
    conv2d_73 (Conv2D)           (None, 128, 128, 32)      9248      
    _________________________________________________________________
    batch_normalization_36 (Batc (None, 128, 128, 32)      128       
    _________________________________________________________________
    max_pooling2d_29 (MaxPooling (None, 64, 64, 32)        0         
    _________________________________________________________________
    conv2d_74 (Conv2D)           (None, 64, 64, 64)        18496     
    _________________________________________________________________
    conv2d_75 (Conv2D)           (None, 64, 64, 64)        36928     
    _________________________________________________________________
    batch_normalization_37 (Batc (None, 64, 64, 64)        256       
    _________________________________________________________________
    max_pooling2d_30 (MaxPooling (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_76 (Conv2D)           (None, 32, 32, 128)       73856     
    _________________________________________________________________
    conv2d_77 (Conv2D)           (None, 32, 32, 128)       147584    
    _________________________________________________________________
    batch_normalization_38 (Batc (None, 32, 32, 128)       512       
    _________________________________________________________________
    global_average_pooling2d_7 ( (None, 128)               0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 128)               16512     
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_25 (Dense)             (None, 16)                2064      
    _________________________________________________________________
    dense_26 (Dense)             (None, 1)                 17        
    =================================================================
    Total params: 313,073
    Trainable params: 312,593
    Non-trainable params: 480
    _________________________________________________________________
    


```python
# This is the same callback system as the second model

thirdModel_filepath = os.path.join('third_model')
thirdModelEarlyStop = EarlyStopping(patience=10, mode = 'auto', restore_best_weights=False, monitor='val_loss')
thirdModelCheckpoint = ModelCheckpoint(thirdModel_filepath,save_best_only=True, monitor='val_loss')
thirdModelLRAdjust = ReduceLROnPlateau(monitor = 'val_loss', factor = .5, patience=2, min_delta=.00000000001)
```


```python
thirdModelHistory,params = create_model_visuals(
    model = thirdModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 100, 
    batch_size=64,
    class_weight = {0:1, 1:1},
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[thirdModelCheckpoint, thirdModelEarlyStop, thirdModelLRAdjust]})
```

    Epoch 1/100
    42/42 [==============================] - ETA: 0s - loss: 0.5296 - accuracy: 0.7535 - true_positives: 849.0000 - true_negatives: 1172.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 40s 955ms/step - loss: 0.5296 - accuracy: 0.7535 - true_positives: 849.0000 - true_negatives: 1172.0000 - val_loss: 0.6609 - val_accuracy: 0.5936 - val_true_positives: 109.0000 - val_true_negatives: 2.0000
    Epoch 2/100
    42/42 [==============================] - 37s 883ms/step - loss: 0.4773 - accuracy: 0.7841 - true_positives: 909.0000 - true_negatives: 1194.0000 - val_loss: 1.0738 - val_accuracy: 0.6257 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 3/100
    42/42 [==============================] - ETA: 0s - loss: 0.4730 - accuracy: 0.7875 - true_positives: 909.0000 - true_negatives: 1203.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 40s 964ms/step - loss: 0.4730 - accuracy: 0.7875 - true_positives: 909.0000 - true_negatives: 1203.0000 - val_loss: 0.4959 - val_accuracy: 0.7273 - val_true_positives: 117.0000 - val_true_negatives: 19.0000
    Epoch 4/100
    42/42 [==============================] - 38s 897ms/step - loss: 0.4679 - accuracy: 0.7886 - true_positives: 886.0000 - true_negatives: 1229.0000 - val_loss: 1.3841 - val_accuracy: 0.3743 - val_true_positives: 0.0000e+00 - val_true_negatives: 70.0000
    Epoch 5/100
    42/42 [==============================] - 38s 904ms/step - loss: 0.4610 - accuracy: 0.7957 - true_positives: 888.0000 - true_negatives: 1246.0000 - val_loss: 0.5029 - val_accuracy: 0.7433 - val_true_positives: 71.0000 - val_true_negatives: 68.0000
    Epoch 6/100
    42/42 [==============================] - 38s 896ms/step - loss: 0.4442 - accuracy: 0.8024 - true_positives: 885.0000 - true_negatives: 1267.0000 - val_loss: 0.6610 - val_accuracy: 0.5615 - val_true_positives: 85.0000 - val_true_negatives: 20.0000
    Epoch 7/100
    42/42 [==============================] - ETA: 0s - loss: 0.4377 - accuracy: 0.8009 - true_positives: 881.0000 - true_negatives: 1267.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 40s 962ms/step - loss: 0.4377 - accuracy: 0.8009 - true_positives: 881.0000 - true_negatives: 1267.0000 - val_loss: 0.3943 - val_accuracy: 0.7914 - val_true_positives: 117.0000 - val_true_negatives: 31.0000
    Epoch 8/100
    42/42 [==============================] - 38s 898ms/step - loss: 0.4352 - accuracy: 0.8054 - true_positives: 889.0000 - true_negatives: 1271.0000 - val_loss: 0.5806 - val_accuracy: 0.6364 - val_true_positives: 49.0000 - val_true_negatives: 70.0000
    Epoch 9/100
    42/42 [==============================] - 38s 906ms/step - loss: 0.4267 - accuracy: 0.8084 - true_positives: 899.0000 - true_negatives: 1269.0000 - val_loss: 0.8439 - val_accuracy: 0.3904 - val_true_positives: 3.0000 - val_true_negatives: 70.0000
    Epoch 10/100
    42/42 [==============================] - ETA: 0s - loss: 0.4226 - accuracy: 0.8162 - true_positives: 896.0000 - true_negatives: 1293.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 40s 962ms/step - loss: 0.4226 - accuracy: 0.8162 - true_positives: 896.0000 - true_negatives: 1293.0000 - val_loss: 0.2847 - val_accuracy: 0.8984 - val_true_positives: 101.0000 - val_true_negatives: 67.0000
    Epoch 11/100
    42/42 [==============================] - 36s 865ms/step - loss: 0.4214 - accuracy: 0.8117 - true_positives: 901.0000 - true_negatives: 1276.0000 - val_loss: 0.3482 - val_accuracy: 0.8824 - val_true_positives: 95.0000 - val_true_negatives: 70.0000
    Epoch 12/100
    42/42 [==============================] - 36s 847ms/step - loss: 0.4200 - accuracy: 0.8147 - true_positives: 906.0000 - true_negatives: 1279.0000 - val_loss: 0.3352 - val_accuracy: 0.8610 - val_true_positives: 91.0000 - val_true_negatives: 70.0000
    Epoch 13/100
    42/42 [==============================] - ETA: 0s - loss: 0.4167 - accuracy: 0.8188 - true_positives: 901.0000 - true_negatives: 1295.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 38s 902ms/step - loss: 0.4167 - accuracy: 0.8188 - true_positives: 901.0000 - true_negatives: 1295.0000 - val_loss: 0.2506 - val_accuracy: 0.9091 - val_true_positives: 104.0000 - val_true_negatives: 66.0000
    Epoch 14/100
    42/42 [==============================] - 36s 847ms/step - loss: 0.4118 - accuracy: 0.8236 - true_positives: 926.0000 - true_negatives: 1283.0000 - val_loss: 0.3157 - val_accuracy: 0.8663 - val_true_positives: 92.0000 - val_true_negatives: 70.0000
    Epoch 15/100
    42/42 [==============================] - ETA: 0s - loss: 0.4160 - accuracy: 0.8225 - true_positives: 914.0000 - true_negatives: 1292.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 38s 897ms/step - loss: 0.4160 - accuracy: 0.8225 - true_positives: 914.0000 - true_negatives: 1292.0000 - val_loss: 0.2483 - val_accuracy: 0.9091 - val_true_positives: 101.0000 - val_true_negatives: 69.0000
    Epoch 16/100
    42/42 [==============================] - 35s 830ms/step - loss: 0.4138 - accuracy: 0.8218 - true_positives: 917.0000 - true_negatives: 1287.0000 - val_loss: 0.2753 - val_accuracy: 0.8877 - val_true_positives: 96.0000 - val_true_negatives: 70.0000
    Epoch 17/100
    42/42 [==============================] - 35s 828ms/step - loss: 0.4131 - accuracy: 0.8233 - true_positives: 915.0000 - true_negatives: 1293.0000 - val_loss: 0.2861 - val_accuracy: 0.8770 - val_true_positives: 95.0000 - val_true_negatives: 69.0000
    Epoch 18/100
    42/42 [==============================] - 35s 828ms/step - loss: 0.4052 - accuracy: 0.8285 - true_positives: 927.0000 - true_negatives: 1295.0000 - val_loss: 0.3726 - val_accuracy: 0.8235 - val_true_positives: 84.0000 - val_true_negatives: 70.0000
    Epoch 19/100
    42/42 [==============================] - 34s 815ms/step - loss: 0.4017 - accuracy: 0.8274 - true_positives: 932.0000 - true_negatives: 1287.0000 - val_loss: 0.3421 - val_accuracy: 0.8396 - val_true_positives: 87.0000 - val_true_negatives: 70.0000
    Epoch 20/100
    42/42 [==============================] - 35s 826ms/step - loss: 0.4034 - accuracy: 0.8248 - true_positives: 951.0000 - true_negatives: 1261.0000 - val_loss: 0.2851 - val_accuracy: 0.8770 - val_true_positives: 94.0000 - val_true_negatives: 70.0000
    Epoch 21/100
    42/42 [==============================] - ETA: 0s - loss: 0.4001 - accuracy: 0.8240 - true_positives: 927.0000 - true_negatives: 1283.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 37s 882ms/step - loss: 0.4001 - accuracy: 0.8240 - true_positives: 927.0000 - true_negatives: 1283.0000 - val_loss: 0.2393 - val_accuracy: 0.9144 - val_true_positives: 103.0000 - val_true_negatives: 68.0000
    Epoch 22/100
    42/42 [==============================] - 35s 827ms/step - loss: 0.4060 - accuracy: 0.8251 - true_positives: 930.0000 - true_negatives: 1283.0000 - val_loss: 0.2432 - val_accuracy: 0.9358 - val_true_positives: 106.0000 - val_true_negatives: 69.0000
    Epoch 23/100
    42/42 [==============================] - ETA: 0s - loss: 0.4126 - accuracy: 0.8214 - true_positives: 924.0000 - true_negatives: 1279.0000INFO:tensorflow:Assets written to: third_model/assets
    42/42 [==============================] - 37s 878ms/step - loss: 0.4126 - accuracy: 0.8214 - true_positives: 924.0000 - true_negatives: 1279.0000 - val_loss: 0.2290 - val_accuracy: 0.9144 - val_true_positives: 103.0000 - val_true_negatives: 68.0000
    Epoch 24/100
    42/42 [==============================] - 34s 814ms/step - loss: 0.4078 - accuracy: 0.8225 - true_positives: 932.0000 - true_negatives: 1274.0000 - val_loss: 0.2896 - val_accuracy: 0.8770 - val_true_positives: 94.0000 - val_true_negatives: 70.0000
    Epoch 25/100
    42/42 [==============================] - 35s 824ms/step - loss: 0.4045 - accuracy: 0.8262 - true_positives: 929.0000 - true_negatives: 1287.0000 - val_loss: 0.2787 - val_accuracy: 0.8877 - val_true_positives: 96.0000 - val_true_negatives: 70.0000
    Epoch 26/100
    42/42 [==============================] - 35s 837ms/step - loss: 0.3995 - accuracy: 0.8255 - true_positives: 933.0000 - true_negatives: 1281.0000 - val_loss: 0.2616 - val_accuracy: 0.8930 - val_true_positives: 98.0000 - val_true_negatives: 69.0000
    Epoch 27/100
    42/42 [==============================] - 35s 834ms/step - loss: 0.4059 - accuracy: 0.8285 - true_positives: 940.0000 - true_negatives: 1282.0000 - val_loss: 0.2579 - val_accuracy: 0.9037 - val_true_positives: 100.0000 - val_true_negatives: 69.0000
    Epoch 28/100
    42/42 [==============================] - 35s 822ms/step - loss: 0.4041 - accuracy: 0.8274 - true_positives: 933.0000 - true_negatives: 1286.0000 - val_loss: 0.2503 - val_accuracy: 0.9091 - val_true_positives: 101.0000 - val_true_negatives: 69.0000
    Epoch 29/100
    42/42 [==============================] - 35s 823ms/step - loss: 0.4066 - accuracy: 0.8203 - true_positives: 919.0000 - true_negatives: 1281.0000 - val_loss: 0.2501 - val_accuracy: 0.9037 - val_true_positives: 100.0000 - val_true_negatives: 69.0000
    Epoch 30/100
    42/42 [==============================] - 34s 819ms/step - loss: 0.4066 - accuracy: 0.8292 - true_positives: 938.0000 - true_negatives: 1286.0000 - val_loss: 0.2472 - val_accuracy: 0.9091 - val_true_positives: 101.0000 - val_true_negatives: 69.0000
    Epoch 31/100
    42/42 [==============================] - 35s 822ms/step - loss: 0.4076 - accuracy: 0.8248 - true_positives: 934.0000 - true_negatives: 1278.0000 - val_loss: 0.2468 - val_accuracy: 0.9144 - val_true_positives: 102.0000 - val_true_negatives: 69.0000
    Epoch 32/100
    42/42 [==============================] - 34s 821ms/step - loss: 0.3980 - accuracy: 0.8311 - true_positives: 940.0000 - true_negatives: 1289.0000 - val_loss: 0.2465 - val_accuracy: 0.9091 - val_true_positives: 101.0000 - val_true_negatives: 69.0000
    Epoch 33/100
    42/42 [==============================] - 34s 821ms/step - loss: 0.4033 - accuracy: 0.8285 - true_positives: 931.0000 - true_negatives: 1291.0000 - val_loss: 0.2468 - val_accuracy: 0.9091 - val_true_positives: 101.0000 - val_true_negatives: 69.0000
    7/7 [==============================] - 0s 54ms/step - loss: 0.4093 - accuracy: 0.8055 - true_positives: 228.0000 - true_negatives: 124.0000
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.733728</td>
      <td>0.850746</td>
      <td>0.805492</td>
      <td>0.792237</td>
      <td>0.806831</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.756098</td>
      <td>0.835165</td>
      <td>0.805492</td>
      <td>0.795631</td>
      <td>0.805492</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.744745</td>
      <td>0.842884</td>
      <td>0.805492</td>
      <td>0.793814</td>
      <td>0.806053</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.805492</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_43_2.png)



![png](output_43_3.png)



```python
keras.models.save_model(thirdModel, 'drive/MyDrive/Colab Notebooks/flatiron school/third_model_non_checkpoint/')
thirdModel = keras.models.load_model('third_model/')
keras.models.save_model(thirdModel, 'drive/MyDrive/Colab Notebooks/flatiron school/third_model_checkpoint/')
```

    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/third_model_non_checkpoint/assets
    


```python
thirdModelHistory,params = create_model_visuals(
    model = thirdModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 100, 
    batch_size=64,
    class_weight = scale_weight(1),
    train = False,
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[thirdModelCheckpoint, thirdModelEarlyStop]})
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.795620</td>
      <td>0.816667</td>
      <td>0.810069</td>
      <td>0.806144</td>
      <td>0.808768</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.664634</td>
      <td>0.897436</td>
      <td>0.810069</td>
      <td>0.781035</td>
      <td>0.810069</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.724252</td>
      <td>0.855148</td>
      <td>0.810069</td>
      <td>0.789700</td>
      <td>0.806025</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.810069</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_45_1.png)


# Fourth Model: Oversampling The Minority Class
## Changes From The Third Model
The fourth model is going to attempt an oversampling of the minority class. In terms of the model architecture, some small tweeks to the class weights will be made but no other hyper-parameters will be changed. 
## Building Fourth Model
### Data Preperation
This time around the name of the game is generating "new" images of the minority class that can be used to supplement. In order to do that we will be placing the existing minority class images into a new Imagedatagenerator object and using augmentation in order to artificially create new images that will then be trained on. The hope of this process is to increase the model's exposure to more x-rays of healthy lungs. Currently, the minority class is regularly preforming well below the majority case. This is despite attempts to weight the classes such that the model places more emphasis on the minority class images when making predictions. 
### Model Architecture
### Model Fitting

<img src = 'fourth-model-training-metrics.png'>

## Fourth Model Evaluation
### General Preformace
The confustion matirx for the fourth model.

<img src = 'fourth-model-confusion-matrix.png'>

The fourth model has preformed better than the baseline but worse than the second model. Our metrics are rough and not dempnstrating the learning curves that instill me with the confidence that the model is learning the underlying structure of the images themselves. What's odd here is the high precision for the negative class. When the model chose to guess the negative class it was doing so with 97% precision. Thus, it was confident of its choice when choosing the negative class. This could represent the additional exposures to the same images with augmentation tuned the model to specific images (memorization) rather then "truely" learning the features of negative images.  

### Accuracy

81% Accuracy overall. This is a surprising decrease from the second model. This deserves more attention (perhaps as future work) as in general, increasing the total amount of data should almost always increase performace. 

### Pneumonia Recall

The recall on the positive case was 98%. This is almost optimal given the task is medical diagnosis and catching the positive case is more important than strict accuracy. 

### Non-Pneumonia Recall

Recall preformace on the negative class was 73%. With a 27% false psitive rate we are beginning to boarder on the territory of whether it should be deployed or not. Ultimately, the high positive case recall still makes the model useful but perhaps not within the scope of expiditing emergancy cases. If there are too many flase positives refered to the radiologists, then the model will only serve to gum up the works and might decrease overall efficiency. 

## Code To Create The Fourth Model


```python
x_p = trainImages[trainImageLabels == 1]
x_pLabelds = [1 for i in range(len(x_p))]

x_n = trainImages[trainImageLabels == 0]
x_nLabelds = [1 for i in range(len(x_n))]


```


```python
i=0
for batch in trainGenerator.flow(
    x_n,
    shuffle = False,
    batch_size = 1,
    save_to_dir = '/content/dsc-mod-4-project-v2-1-onl01-dtsc-pt-041320/data/chest_xray/train/NORMAL',
    save_prefix = 'class0'
):
  i += 1
  if i > 2500:
    break
```


```python
# Create an image generator for the training set. We are spliting a validation set off of this training set
trainGenerator0 = ImageDataGenerator(
    rescale = 1./255
)
 
# Create an image generator for the training set. We are spliting a validation set off of this training set
testGenerator0 = ImageDataGenerator(
    validation_split=.3,
    rescale = 1./255
)
```


```python
# The flowed data from the directory for the training set. This will be fed directly into the neural networks
train0 = trainGenerator0.flow_from_directory(
    trainPath,  
    class_mode='binary',
    target_size=(256,256),
    batch_size = 7717,
    shuffle = True)
 
# The split off validation set. This is fed directly into the neural networks
val0 = testGenerator0.flow_from_directory(
    testPath, 
    class_mode='binary',
    target_size=(256,256),
    batch_size = 187,
    subset='validation',
    shuffle = True
)
 
 
# The flowed data from the directory for the testing set. This will be fed directly into the neural networks
test0 = testGenerator0.flow_from_directory(
    testPath, 
    class_mode='binary',
    target_size=(256,256),
    batch_size = 437,
    subset = 'training',
    shuffle = True
)
```

    Found 7717 images belonging to 2 classes.
    Found 187 images belonging to 2 classes.
    Found 437 images belonging to 2 classes.
    


```python
trainImages, trainImageLabels = next(train0)
valImages, valImageLabels = next(val0)
testImages, testImageLabels = next(test0)
```


```python
# The flowed data from the top of the notebook. This will be fed directly into the neural networks
 
train = trainGenerator.flow(
    trainImages,
    y = trainImageLabels, 
    shuffle = False,
    batch_size = 64,
)
# The split off validation set. This is fed directly into the neural networks
val = testGenerator.flow(
    valImages,
    y = valImageLabels, 
    shuffle = False,
    batch_size = 64,
 
)
 
 
# The flowed data from the top of the notebook. This will be fed directly into the neural networks
test = testGenerator.flow(
    testImages,
    y = testImageLabels, 
    shuffle = False,
    batch_size = 64,
 
)
```


```python
fourthModel = Sequential()
fourthModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      input_shape = (256,256,3)
      )
    )
fourthModel.add(
    layers.Conv2D(
      16,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
fourthModel.add(layers.MaxPooling2D(2))
 
fourthModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
fourthModel.add(
    layers.Conv2D(
      32,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
fourthModel.add(layers.MaxPooling2D(2))
 
fourthModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
fourthModel.add(
    layers.Conv2D(
      64,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
 
fourthModel.add(layers.MaxPooling2D(2))
 
 
fourthModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
fourthModel.add(
    layers.Conv2D(
      128,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
 
fourthModel.add(layers.MaxPooling2D(2))
 
fourthModel.add(
    layers.Conv2D(
      256,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
fourthModel.add(
    layers.Conv2D(
      256,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
 
fourthModel.add(layers.MaxPooling2D(2))
 
fourthModel.add(
    layers.Conv2D(
      512,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      kernel_initializer=keras.initializers.Orthogonal(), 
      )
    )
 
fourthModel.add(
    layers.Conv2D(
      512,
      strides = 1,
      kernel_size = (3,3),
      activation = 'relu', 
      padding = 'same', 
      )
    )
 
fourthModel.add(layers.BatchNormalization())
 
fourthModel.add(layers.GlobalAveragePooling2D())
 
 
fourthModel.add(layers.Dense(
    128, 
    kernel_initializer=keras.initializers.Orthogonal(),
    activity_regularizer = keras.regularizers.L2(.02),
    activation = 'relu'
    )
)
 
 
fourthModel.add(layers.BatchNormalization())
 
fourthModel.add(layers.Dense(
    6,
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal()
    )
)
 
 
fourthModel.add(layers.Dense(
    6, 
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal(),
    ))
 
fourthModel.add(layers.Dense(1, activation = 'sigmoid'))
 
 
fourthModel.compile(
    optimizer = keras.optimizers.RMSprop(
        learning_rate=.001,
        # momentum = .01
        ),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [
      keras.metrics.BinaryAccuracy(name = 'accuracy'),
      keras.metrics.AUC(name = 'auc'),
      keras.metrics.SquaredHinge(name = 'square_hinge'),
      keras.metrics.TruePositives(name='true_positives'), 
      keras.metrics.TrueNegatives(name = 'true_negatives')]
)
fourthModel.summary()
```

    Model: "sequential_20"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_240 (Conv2D)          (None, 256, 256, 16)      448       
    _________________________________________________________________
    conv2d_241 (Conv2D)          (None, 256, 256, 16)      2320      
    _________________________________________________________________
    batch_normalization_155 (Bat (None, 256, 256, 16)      64        
    _________________________________________________________________
    max_pooling2d_100 (MaxPoolin (None, 128, 128, 16)      0         
    _________________________________________________________________
    conv2d_242 (Conv2D)          (None, 128, 128, 32)      4640      
    _________________________________________________________________
    conv2d_243 (Conv2D)          (None, 128, 128, 32)      9248      
    _________________________________________________________________
    batch_normalization_156 (Bat (None, 128, 128, 32)      128       
    _________________________________________________________________
    max_pooling2d_101 (MaxPoolin (None, 64, 64, 32)        0         
    _________________________________________________________________
    conv2d_244 (Conv2D)          (None, 64, 64, 64)        18496     
    _________________________________________________________________
    conv2d_245 (Conv2D)          (None, 64, 64, 64)        36928     
    _________________________________________________________________
    batch_normalization_157 (Bat (None, 64, 64, 64)        256       
    _________________________________________________________________
    max_pooling2d_102 (MaxPoolin (None, 32, 32, 64)        0         
    _________________________________________________________________
    conv2d_246 (Conv2D)          (None, 32, 32, 128)       73856     
    _________________________________________________________________
    conv2d_247 (Conv2D)          (None, 32, 32, 128)       147584    
    _________________________________________________________________
    batch_normalization_158 (Bat (None, 32, 32, 128)       512       
    _________________________________________________________________
    max_pooling2d_103 (MaxPoolin (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_248 (Conv2D)          (None, 16, 16, 256)       295168    
    _________________________________________________________________
    conv2d_249 (Conv2D)          (None, 16, 16, 256)       590080    
    _________________________________________________________________
    batch_normalization_159 (Bat (None, 16, 16, 256)       1024      
    _________________________________________________________________
    max_pooling2d_104 (MaxPoolin (None, 8, 8, 256)         0         
    _________________________________________________________________
    conv2d_250 (Conv2D)          (None, 8, 8, 512)         1180160   
    _________________________________________________________________
    conv2d_251 (Conv2D)          (None, 8, 8, 512)         2359808   
    _________________________________________________________________
    batch_normalization_160 (Bat (None, 8, 8, 512)         2048      
    _________________________________________________________________
    global_average_pooling2d_20  (None, 512)               0         
    _________________________________________________________________
    dense_80 (Dense)             (None, 128)               65664     
    _________________________________________________________________
    batch_normalization_161 (Bat (None, 128)               512       
    _________________________________________________________________
    dense_81 (Dense)             (None, 6)                 774       
    _________________________________________________________________
    dense_82 (Dense)             (None, 6)                 42        
    _________________________________________________________________
    dense_83 (Dense)             (None, 1)                 7         
    =================================================================
    Total params: 4,789,767
    Trainable params: 4,787,495
    Non-trainable params: 2,272
    _________________________________________________________________
    


```python
# This cell defines the callback we will use for the fit method.

# we define the filepath where any results will be saved
fourthModel_filepath = os.path.join('fourth_model')

# the Callbacks
fourthModelEarlyStop = EarlyStopping(patience= 8, mode = 'auto', restore_best_weights=False, monitor='val_loss')
fourthModelCheckpoint = ModelCheckpoint(fourthModel_filepath,save_best_only=True, monitor='val_loss')
fourthModelLRAdjust = ReduceLROnPlateau(monitor = 'val_loss', factor = .5, patience=2, min_delta=.00000000001)
```


```python
# This cell runs the fitting 
fourthModelHistory,params = create_model_visuals(
    model = fourthModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator = test, 
    epochs = 100, 
    batch_size=64,
    class_weight = {0:1, 1:1},
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[fourthModelCheckpoint, fourthModelEarlyStop, fourthModelLRAdjust]}
        )
```

    Epoch 1/100
    121/121 [==============================] - ETA: 0s - loss: 0.4346 - accuracy: 0.8424 - auc: 0.9118 - square_hinge: 0.8387 - true_positives: 3127.0000 - true_negatives: 3374.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 100s 827ms/step - loss: 0.4346 - accuracy: 0.8424 - auc: 0.9118 - square_hinge: 0.8387 - true_positives: 3127.0000 - true_negatives: 3374.0000 - val_loss: 1.6541 - val_accuracy: 0.6257 - val_auc: 0.3849 - val_square_hinge: 1.3115 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 2/100
    121/121 [==============================] - 96s 790ms/step - loss: 0.2842 - accuracy: 0.8991 - auc: 0.9615 - square_hinge: 0.7331 - true_positives: 3443.0000 - true_negatives: 3495.0000 - val_loss: 1.7794 - val_accuracy: 0.6257 - val_auc: 0.5594 - val_square_hinge: 1.4385 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 3/100
    121/121 [==============================] - ETA: 0s - loss: 0.2182 - accuracy: 0.9260 - auc: 0.9771 - square_hinge: 0.6734 - true_positives: 3544.0000 - true_negatives: 3602.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 99s 819ms/step - loss: 0.2182 - accuracy: 0.9260 - auc: 0.9771 - square_hinge: 0.6734 - true_positives: 3544.0000 - true_negatives: 3602.0000 - val_loss: 1.5909 - val_accuracy: 0.6257 - val_auc: 0.4872 - val_square_hinge: 1.4646 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 4/100
    121/121 [==============================] - ETA: 0s - loss: 0.1956 - accuracy: 0.9343 - auc: 0.9806 - square_hinge: 0.6530 - true_positives: 3593.0000 - true_negatives: 3617.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 101s 837ms/step - loss: 0.1956 - accuracy: 0.9343 - auc: 0.9806 - square_hinge: 0.6530 - true_positives: 3593.0000 - true_negatives: 3617.0000 - val_loss: 0.9466 - val_accuracy: 0.6631 - val_auc: 0.8922 - val_square_hinge: 1.2548 - val_true_positives: 117.0000 - val_true_negatives: 7.0000
    Epoch 5/100
    121/121 [==============================] - 96s 793ms/step - loss: 0.1712 - accuracy: 0.9391 - auc: 0.9845 - square_hinge: 0.6345 - true_positives: 3603.0000 - true_negatives: 3644.0000 - val_loss: 1.3779 - val_accuracy: 0.6257 - val_auc: 0.9189 - val_square_hinge: 1.4453 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 6/100
    121/121 [==============================] - ETA: 0s - loss: 0.1527 - accuracy: 0.9502 - auc: 0.9872 - square_hinge: 0.6163 - true_positives: 3656.0000 - true_negatives: 3677.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 99s 814ms/step - loss: 0.1527 - accuracy: 0.9502 - auc: 0.9872 - square_hinge: 0.6163 - true_positives: 3656.0000 - true_negatives: 3677.0000 - val_loss: 0.3649 - val_accuracy: 0.8021 - val_auc: 0.9653 - val_square_hinge: 0.5712 - val_true_positives: 83.0000 - val_true_negatives: 67.0000
    Epoch 7/100
    121/121 [==============================] - 95s 786ms/step - loss: 0.1465 - accuracy: 0.9519 - auc: 0.9885 - square_hinge: 0.6144 - true_positives: 3682.0000 - true_negatives: 3664.0000 - val_loss: 0.8200 - val_accuracy: 0.6310 - val_auc: 0.9606 - val_square_hinge: 1.2935 - val_true_positives: 117.0000 - val_true_negatives: 1.0000
    Epoch 8/100
    121/121 [==============================] - 94s 779ms/step - loss: 0.1398 - accuracy: 0.9532 - auc: 0.9896 - square_hinge: 0.6076 - true_positives: 3671.0000 - true_negatives: 3685.0000 - val_loss: 0.3975 - val_accuracy: 0.8235 - val_auc: 0.9833 - val_square_hinge: 0.8559 - val_true_positives: 117.0000 - val_true_negatives: 37.0000
    Epoch 9/100
    121/121 [==============================] - ETA: 0s - loss: 0.1067 - accuracy: 0.9638 - auc: 0.9937 - square_hinge: 0.5840 - true_positives: 3718.0000 - true_negatives: 3720.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 97s 802ms/step - loss: 0.1067 - accuracy: 0.9638 - auc: 0.9937 - square_hinge: 0.5840 - true_positives: 3718.0000 - true_negatives: 3720.0000 - val_loss: 0.1202 - val_accuracy: 0.9626 - val_auc: 0.9932 - val_square_hinge: 0.4983 - val_true_positives: 117.0000 - val_true_negatives: 63.0000
    Epoch 10/100
    121/121 [==============================] - 95s 781ms/step - loss: 0.0961 - accuracy: 0.9685 - auc: 0.9947 - square_hinge: 0.5767 - true_positives: 3743.0000 - true_negatives: 3731.0000 - val_loss: 0.2880 - val_accuracy: 0.9144 - val_auc: 0.9769 - val_square_hinge: 0.6619 - val_true_positives: 117.0000 - val_true_negatives: 54.0000
    Epoch 11/100
    121/121 [==============================] - ETA: 0s - loss: 0.0928 - accuracy: 0.9708 - auc: 0.9946 - square_hinge: 0.5705 - true_positives: 3750.0000 - true_negatives: 3742.0000INFO:tensorflow:Assets written to: fourth_model/assets
    121/121 [==============================] - 99s 814ms/step - loss: 0.0928 - accuracy: 0.9708 - auc: 0.9946 - square_hinge: 0.5705 - true_positives: 3750.0000 - true_negatives: 3742.0000 - val_loss: 0.0932 - val_accuracy: 0.9733 - val_auc: 0.9906 - val_square_hinge: 0.4704 - val_true_positives: 115.0000 - val_true_negatives: 67.0000
    Epoch 12/100
    121/121 [==============================] - 95s 786ms/step - loss: 0.0885 - accuracy: 0.9718 - auc: 0.9953 - square_hinge: 0.5658 - true_positives: 3757.0000 - true_negatives: 3742.0000 - val_loss: 0.1368 - val_accuracy: 0.9626 - val_auc: 0.9836 - val_square_hinge: 0.5080 - val_true_positives: 115.0000 - val_true_negatives: 65.0000
    Epoch 13/100
    121/121 [==============================] - 95s 786ms/step - loss: 0.0815 - accuracy: 0.9724 - auc: 0.9958 - square_hinge: 0.5624 - true_positives: 3763.0000 - true_negatives: 3741.0000 - val_loss: 0.2746 - val_accuracy: 0.8877 - val_auc: 0.9824 - val_square_hinge: 0.7184 - val_true_positives: 116.0000 - val_true_negatives: 50.0000
    Epoch 14/100
    121/121 [==============================] - 95s 784ms/step - loss: 0.0609 - accuracy: 0.9825 - auc: 0.9977 - square_hinge: 0.5473 - true_positives: 3805.0000 - true_negatives: 3777.0000 - val_loss: 0.1135 - val_accuracy: 0.9679 - val_auc: 0.9957 - val_square_hinge: 0.4999 - val_true_positives: 117.0000 - val_true_negatives: 64.0000
    Epoch 15/100
    121/121 [==============================] - 95s 784ms/step - loss: 0.0643 - accuracy: 0.9816 - auc: 0.9971 - square_hinge: 0.5451 - true_positives: 3789.0000 - true_negatives: 3786.0000 - val_loss: 0.6148 - val_accuracy: 0.8075 - val_auc: 0.9877 - val_square_hinge: 0.9674 - val_true_positives: 117.0000 - val_true_negatives: 34.0000
    Epoch 16/100
    121/121 [==============================] - 94s 775ms/step - loss: 0.0480 - accuracy: 0.9854 - auc: 0.9986 - square_hinge: 0.5367 - true_positives: 3813.0000 - true_negatives: 3791.0000 - val_loss: 0.4005 - val_accuracy: 0.8717 - val_auc: 0.9874 - val_square_hinge: 0.7818 - val_true_positives: 117.0000 - val_true_negatives: 46.0000
    Epoch 17/100
    121/121 [==============================] - 94s 779ms/step - loss: 0.0482 - accuracy: 0.9852 - auc: 0.9989 - square_hinge: 0.5366 - true_positives: 3811.0000 - true_negatives: 3792.0000 - val_loss: 0.3228 - val_accuracy: 0.9091 - val_auc: 0.9898 - val_square_hinge: 0.6526 - val_true_positives: 117.0000 - val_true_negatives: 53.0000
    Epoch 18/100
    121/121 [==============================] - 94s 774ms/step - loss: 0.0425 - accuracy: 0.9876 - auc: 0.9989 - square_hinge: 0.5333 - true_positives: 3819.0000 - true_negatives: 3802.0000 - val_loss: 0.3691 - val_accuracy: 0.8930 - val_auc: 0.9825 - val_square_hinge: 0.6990 - val_true_positives: 117.0000 - val_true_negatives: 50.0000
    Epoch 19/100
    121/121 [==============================] - 95s 782ms/step - loss: 0.0456 - accuracy: 0.9848 - auc: 0.9987 - square_hinge: 0.5344 - true_positives: 3810.0000 - true_negatives: 3790.0000 - val_loss: 0.4324 - val_accuracy: 0.8770 - val_auc: 0.9829 - val_square_hinge: 0.7432 - val_true_positives: 117.0000 - val_true_negatives: 47.0000
    7/7 [==============================] - 0s 44ms/step - loss: 1.2527 - accuracy: 0.6934 - auc: 0.9182 - square_hinge: 1.2559 - true_positives: 273.0000 - true_negatives: 30.0000
    


![png](output_56_1.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>1.000000</td>
      <td>0.670762</td>
      <td>0.693364</td>
      <td>0.835381</td>
      <td>0.794320</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.182927</td>
      <td>1.000000</td>
      <td>0.693364</td>
      <td>0.591463</td>
      <td>0.693364</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.309278</td>
      <td>0.802941</td>
      <td>0.693364</td>
      <td>0.556110</td>
      <td>0.617676</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.693364</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_56_3.png)



```python
# The previous fitting method saves the model with the least validation loss score, but also prints out
# the model when the early stopping callback stops the fitting. Because of this we get two models from the previous cell

# save the model spit out by the fit method
keras.models.save_model(fourthModel, 'drive/MyDrive/Colab Notebooks/flatiron school/fourth_model_non_checkpoint/')
#load the model saved by the checkpoint callback
fourthModel = keras.models.load_model('fourth_model/')
#show the eval metrics for the check pointed model
keras.models.save_model(fourthModel, 'drive/MyDrive/Colab Notebooks/flatiron school/fourth_model_checkpoint/')

fourthModel = keras.models.load_model('drive/MyDrive/Colab Notebooks/flatiron school/fourth_model_checkpoint/')
fourthModelHistory,params = create_model_visuals(
    model = fourthModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 100, 
    batch_size=64,
    class_weight = scale_weight(1.5),
    train = False,
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[fourthModelCheckpoint, fourthModelEarlyStop]})
```

    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/fourth_model_non_checkpoint/assets
    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/fourth_model_checkpoint/assets
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.960000</td>
      <td>0.858974</td>
      <td>0.887872</td>
      <td>0.909487</td>
      <td>0.896888</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.731707</td>
      <td>0.981685</td>
      <td>0.887872</td>
      <td>0.856696</td>
      <td>0.887872</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.830450</td>
      <td>0.916239</td>
      <td>0.887872</td>
      <td>0.873345</td>
      <td>0.884044</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.887872</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_57_2.png)


# Third Party Benchmark
So we've gone through the trouble of creating a model but what does this mean in the grand scheme of things. Is the model good enough to compete in the data science industry? Are there other existing models that can vastly out perform what we've created?

In this case we can take these open ended questions and get some reasonable answer to them. Keras and several other popular neural network modules offer some pre-trained models whose architectures led them to victory in recent global challenges. One such architecture specifically trained on images is the VGG19 network. At this point we are going to load in this original model and play around with it as a more robust comparison of how our model is preforming within the industry. 

This process has several built in options. The model can come with all the layers from start to finish including both convulutional and fully connected layers already trained. There's also options to take only the connvulutional section pre-trained or to import the model architecture without any pre-training whatsoever. In this case, we will take the connvulutional layers having already been pre-trained as re-training them will take too long without considerable computing reasorces. We will create our own dense layers to add ot this model which will taylor the connvulutional layers for our specific task of classifying x-rays. While it seems odd to take a pre-trained state of the art model and add our own layers to it, because the underlying convulutional models are trained on a hugh variety of images, we need a fully connected architecture that will take the vast knowledge contained in the convulutional layers and funnel it down in a way that maximized our classification task. Unfortuneately, there's currently not a fantastic general structure that succeeds at any given task so there will be some trial and error. 


```python
from keras.applications import VGG19
cnn_base = VGG19(weights='imagenet', 
                 include_top=False, 
                 input_shape=(256, 256, 3))
```


```python
benchmarkModel = Sequential()

benchmarkModel.add(cnn_base)

benchmarkModel.add(layers.BatchNormalization())
 
benchmarkModel.add(layers.GlobalAveragePooling2D())
 
benchmarkModel.add(layers.Dense(
    128, 
    kernel_initializer=keras.initializers.Orthogonal(),
    activity_regularizer = keras.regularizers.L2(.03),
    )
)
 
benchmarkModel.add(layers.BatchNormalization())
 
benchmarkModel.add(layers.Dense(
    6,
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal())
)
 
benchmarkModel.add(layers.BatchNormalization())
 
benchmarkModel.add(layers.Dense(
    6, 
    activation= 'tanh',
    kernel_initializer = keras.initializers.Orthogonal(),
    ))
 
benchmarkModel.add(layers.Dense(1, activation = 'sigmoid'))
 
benchmarkModel.compile(
    optimizer = keras.optimizers.RMSprop(
        learning_rate=.001,
        rho = .9,),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [
      keras.metrics.BinaryAccuracy(name = 'accuracy'),
      keras.metrics.AUC(name = 'auc'),
      keras.metrics.SquaredHinge(name = 'square_hinge'),
      keras.metrics.TruePositives(name='true_positives'), 
      keras.metrics.TrueNegatives(name = 'true_negatives')]
)


# we define the filepath where any results will be saved
benchmarkModel_filepath = os.path.join('benchmark_model')

# the Callbacks
benchmarkModelEarlyStop = EarlyStopping(patience= 7, mode = 'auto', restore_best_weights=False, monitor='val_loss')
benchmarkModelCheckpoint = ModelCheckpoint(benchmarkModel_filepath,save_best_only=True, monitor='val_loss')
benchmarkModelLRAdjust = ReduceLROnPlateau(monitor = 'val_loss', factor = .5, patience=2, min_delta=.00000000001)

cnn_base.trainable = False
benchmarkModel.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg19 (Functional)           (None, 8, 8, 512)         20024384  
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 8, 8, 512)         2048      
    _________________________________________________________________
    global_average_pooling2d_6 ( (None, 512)               0         
    _________________________________________________________________
    dense_26 (Dense)             (None, 128)               65664     
    _________________________________________________________________
    batch_normalization_19 (Batc (None, 128)               512       
    _________________________________________________________________
    dense_27 (Dense)             (None, 6)                 774       
    _________________________________________________________________
    batch_normalization_20 (Batc (None, 6)                 24        
    _________________________________________________________________
    dense_28 (Dense)             (None, 6)                 42        
    _________________________________________________________________
    dense_29 (Dense)             (None, 1)                 7         
    =================================================================
    Total params: 20,093,455
    Trainable params: 67,779
    Non-trainable params: 20,025,676
    _________________________________________________________________
    


```python
# This cell runs the fitting 
benchmarkModelHistory,params = create_model_visuals(
    model = benchmarkModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 150, 
    batch_size=64,
    class_weight = scale_weight(1),
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[benchmarkModelCheckpoint, benchmarkModelEarlyStop, benchmarkModelLRAdjust]})
```

    Epoch 1/150
    82/82 [==============================] - ETA: 0s - loss: 0.3255 - accuracy: 0.8790 - auc: 0.9692 - square_hinge: 0.4672 - true_positives: 3272.0000 - true_negatives: 1313.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 65s 793ms/step - loss: 0.3255 - accuracy: 0.8790 - auc: 0.9692 - square_hinge: 0.4672 - true_positives: 3272.0000 - true_negatives: 1313.0000 - val_loss: 2.6529 - val_accuracy: 0.3743 - val_auc: 0.9547 - val_square_hinge: 0.9477 - val_true_positives: 0.0000e+00 - val_true_negatives: 70.0000
    Epoch 2/150
    82/82 [==============================] - ETA: 0s - loss: 0.2080 - accuracy: 0.9151 - auc: 0.9773 - square_hinge: 0.3800 - true_positives: 3479.0000 - true_negatives: 1294.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 65s 793ms/step - loss: 0.2080 - accuracy: 0.9151 - auc: 0.9773 - square_hinge: 0.3800 - true_positives: 3479.0000 - true_negatives: 1294.0000 - val_loss: 1.1072 - val_accuracy: 0.3743 - val_auc: 0.9853 - val_square_hinge: 0.8485 - val_true_positives: 0.0000e+00 - val_true_negatives: 70.0000
    Epoch 3/150
    82/82 [==============================] - ETA: 0s - loss: 0.1786 - accuracy: 0.9285 - auc: 0.9809 - square_hinge: 0.3609 - true_positives: 3549.0000 - true_negatives: 1294.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 64s 782ms/step - loss: 0.1786 - accuracy: 0.9285 - auc: 0.9809 - square_hinge: 0.3609 - true_positives: 3549.0000 - true_negatives: 1294.0000 - val_loss: 0.5167 - val_accuracy: 0.8930 - val_auc: 0.9835 - val_square_hinge: 0.7653 - val_true_positives: 97.0000 - val_true_negatives: 70.0000
    Epoch 4/150
    82/82 [==============================] - ETA: 0s - loss: 0.1769 - accuracy: 0.9298 - auc: 0.9802 - square_hinge: 0.3586 - true_positives: 3565.0000 - true_negatives: 1285.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 65s 793ms/step - loss: 0.1769 - accuracy: 0.9298 - auc: 0.9802 - square_hinge: 0.3586 - true_positives: 3565.0000 - true_negatives: 1285.0000 - val_loss: 0.2754 - val_accuracy: 0.9305 - val_auc: 0.9864 - val_square_hinge: 0.5930 - val_true_positives: 107.0000 - val_true_negatives: 67.0000
    Epoch 5/150
    82/82 [==============================] - 61s 748ms/step - loss: 0.1761 - accuracy: 0.9321 - auc: 0.9795 - square_hinge: 0.3627 - true_positives: 3576.0000 - true_negatives: 1286.0000 - val_loss: 0.6158 - val_accuracy: 0.6257 - val_auc: 0.9846 - val_square_hinge: 1.1730 - val_true_positives: 117.0000 - val_true_negatives: 0.0000e+00
    Epoch 6/150
    82/82 [==============================] - ETA: 0s - loss: 0.1682 - accuracy: 0.9335 - auc: 0.9819 - square_hinge: 0.3528 - true_positives: 3582.0000 - true_negatives: 1287.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 64s 782ms/step - loss: 0.1682 - accuracy: 0.9335 - auc: 0.9819 - square_hinge: 0.3528 - true_positives: 3582.0000 - true_negatives: 1287.0000 - val_loss: 0.2036 - val_accuracy: 0.9358 - val_auc: 0.9853 - val_square_hinge: 0.6196 - val_true_positives: 115.0000 - val_true_negatives: 60.0000
    Epoch 7/150
    82/82 [==============================] - 61s 744ms/step - loss: 0.1719 - accuracy: 0.9298 - auc: 0.9797 - square_hinge: 0.3575 - true_positives: 3568.0000 - true_negatives: 1282.0000 - val_loss: 0.2367 - val_accuracy: 0.9091 - val_auc: 0.9832 - val_square_hinge: 0.6868 - val_true_positives: 115.0000 - val_true_negatives: 55.0000
    Epoch 8/150
    82/82 [==============================] - 61s 741ms/step - loss: 0.1638 - accuracy: 0.9323 - auc: 0.9827 - square_hinge: 0.3527 - true_positives: 3578.0000 - true_negatives: 1285.0000 - val_loss: 0.2071 - val_accuracy: 0.9465 - val_auc: 0.9833 - val_square_hinge: 0.6178 - val_true_positives: 112.0000 - val_true_negatives: 65.0000
    Epoch 9/150
    82/82 [==============================] - ETA: 0s - loss: 0.1735 - accuracy: 0.9291 - auc: 0.9803 - square_hinge: 0.3617 - true_positives: 3569.0000 - true_negatives: 1277.0000INFO:tensorflow:Assets written to: benchmark_model/assets
    82/82 [==============================] - 64s 777ms/step - loss: 0.1735 - accuracy: 0.9291 - auc: 0.9803 - square_hinge: 0.3617 - true_positives: 3569.0000 - true_negatives: 1277.0000 - val_loss: 0.1794 - val_accuracy: 0.9251 - val_auc: 0.9831 - val_square_hinge: 0.5120 - val_true_positives: 106.0000 - val_true_negatives: 67.0000
    Epoch 10/150
    82/82 [==============================] - 61s 743ms/step - loss: 0.1608 - accuracy: 0.9333 - auc: 0.9837 - square_hinge: 0.3533 - true_positives: 3584.0000 - true_negatives: 1284.0000 - val_loss: 0.2179 - val_accuracy: 0.9037 - val_auc: 0.9841 - val_square_hinge: 0.4816 - val_true_positives: 102.0000 - val_true_negatives: 67.0000
    Epoch 11/150
    82/82 [==============================] - 61s 745ms/step - loss: 0.1645 - accuracy: 0.9289 - auc: 0.9826 - square_hinge: 0.3564 - true_positives: 3568.0000 - true_negatives: 1277.0000 - val_loss: 0.2173 - val_accuracy: 0.8984 - val_auc: 0.9809 - val_square_hinge: 0.4887 - val_true_positives: 101.0000 - val_true_negatives: 67.0000
    Epoch 12/150
    82/82 [==============================] - 61s 739ms/step - loss: 0.1532 - accuracy: 0.9367 - auc: 0.9847 - square_hinge: 0.3496 - true_positives: 3590.0000 - true_negatives: 1296.0000 - val_loss: 0.2360 - val_accuracy: 0.8930 - val_auc: 0.9797 - val_square_hinge: 0.4812 - val_true_positives: 100.0000 - val_true_negatives: 67.0000
    Epoch 13/150
    82/82 [==============================] - 61s 742ms/step - loss: 0.1580 - accuracy: 0.9331 - auc: 0.9835 - square_hinge: 0.3528 - true_positives: 3581.0000 - true_negatives: 1286.0000 - val_loss: 0.1919 - val_accuracy: 0.9198 - val_auc: 0.9814 - val_square_hinge: 0.4950 - val_true_positives: 105.0000 - val_true_negatives: 67.0000
    Epoch 14/150
    82/82 [==============================] - 61s 740ms/step - loss: 0.1566 - accuracy: 0.9335 - auc: 0.9839 - square_hinge: 0.3512 - true_positives: 3585.0000 - true_negatives: 1284.0000 - val_loss: 0.2324 - val_accuracy: 0.9144 - val_auc: 0.9778 - val_square_hinge: 0.6673 - val_true_positives: 114.0000 - val_true_negatives: 57.0000
    Epoch 15/150
    82/82 [==============================] - 61s 741ms/step - loss: 0.1548 - accuracy: 0.9344 - auc: 0.9839 - square_hinge: 0.3528 - true_positives: 3587.0000 - true_negatives: 1287.0000 - val_loss: 0.2239 - val_accuracy: 0.8984 - val_auc: 0.9823 - val_square_hinge: 0.4873 - val_true_positives: 101.0000 - val_true_negatives: 67.0000
    Epoch 16/150
    82/82 [==============================] - 61s 746ms/step - loss: 0.1521 - accuracy: 0.9344 - auc: 0.9846 - square_hinge: 0.3493 - true_positives: 3583.0000 - true_negatives: 1291.0000 - val_loss: 0.2133 - val_accuracy: 0.9091 - val_auc: 0.9810 - val_square_hinge: 0.4882 - val_true_positives: 103.0000 - val_true_negatives: 67.0000
    7/7 [==============================] - 1s 154ms/step - loss: 0.4579 - accuracy: 0.8101 - auc: 0.8824 - square_hinge: 0.7672 - true_positives: 241.0000 - true_negatives: 113.0000
    


![png](output_61_1.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.779310</td>
      <td>0.825342</td>
      <td>0.810069</td>
      <td>0.802326</td>
      <td>0.808067</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.689024</td>
      <td>0.882784</td>
      <td>0.810069</td>
      <td>0.785904</td>
      <td>0.810069</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.731392</td>
      <td>0.853097</td>
      <td>0.810069</td>
      <td>0.792244</td>
      <td>0.807423</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.810069</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_61_3.png)



```python
# The previous fitting method saves the model with the least validation loss score, but also prints out
# the model when the early stopping callback stops the fitting. Because of this we get two models from the previous cell

# save the model spit out by the fit method
keras.models.save_model(benchmarkModel, 'drive/MyDrive/Colab Notebooks/flatiron school/benchmark_model_non_checkpoint/')
#load the model saved by the checkpoint callback
benchmarkModel = keras.models.load_model('benchmark_model/')
#show the eval metrics for the check pointed model
keras.models.save_model(benchmarkModel, 'drive/MyDrive/Colab Notebooks/flatiron school/benchmark_model_checkpoint/')

benchmarkModel = keras.models.load_model('drive/MyDrive/Colab Notebooks/flatiron school/benchmark_model_checkpoint/')
benchmarkModelHistory,params = create_model_visuals(
    model = benchmarkModel, 
    trainGenerator=train, 
    valGenerator=val, 
    testGenerator=test, 
    epochs = 100, 
    batch_size=64,
    class_weight = scale_weight(1),
    train = False,
    kwargs = {
        'validation_data': (valImages, valImageLabels),
        'callbacks':[benchmarkModelCheckpoint, benchmarkModelEarlyStop]})
```

    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/benchmark_model_non_checkpoint/assets
    INFO:tensorflow:Assets written to: drive/MyDrive/Colab Notebooks/flatiron school/benchmark_model_checkpoint/assets
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>accuracy</th>
      <th>macro avg</th>
      <th>weighted avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>precision</th>
      <td>0.838983</td>
      <td>0.796238</td>
      <td>0.80778</td>
      <td>0.817611</td>
      <td>0.812280</td>
    </tr>
    <tr>
      <th>recall</th>
      <td>0.603659</td>
      <td>0.930403</td>
      <td>0.80778</td>
      <td>0.767031</td>
      <td>0.807780</td>
    </tr>
    <tr>
      <th>f1-score</th>
      <td>0.702128</td>
      <td>0.858108</td>
      <td>0.80778</td>
      <td>0.780118</td>
      <td>0.799571</td>
    </tr>
    <tr>
      <th>support</th>
      <td>164.000000</td>
      <td>273.000000</td>
      <td>0.80778</td>
      <td>437.000000</td>
      <td>437.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_62_2.png)


# Experimental Modles
This section contains various modeling structures which were considered but ultimately not chosen. They are included here just for reference.

## Residual Net


```python
inputs = Input(shape = (128,128,3), name = 'input_layer')
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(inputs)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

resid1 = layers.Conv2D(20,kernel_size = (3,3),padding = 'same', strides = 2)(x)
x = layers.BatchNormalization(axis = 3, scale = False)(resid1)
x = layers.LeakyReLU(.3)(resid1)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same', strides = 1, kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same', strides=1)(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

block1 = layers.add([resid1,x])
resid2 = layers.Conv2D(20,kernel_size = (3,3),padding = 'same', strides = 2, kernel_regularizer = 'l2')(block1)
x = layers.BatchNormalization(axis = 3, scale = False)(resid2)
x = layers.LeakyReLU(.3)(x)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 1, kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

block2 = layers.add([x, resid2])
resid3 = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 2)(block2)
x = layers.BatchNormalization(axis = 3, scale = False)(resid3)
x = layers.LeakyReLU(.3)(resid3)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 1)(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

block3 = layers.add([x, resid3])
resid4 = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 2)(block3)
x = layers.BatchNormalization(axis = 3, scale = False)(resid4)
x = layers.LeakyReLU(.3)(resid4)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 1)(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

block4 = layers.add([x, resid4])
resid5 = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 2)(block4)
x = layers.BatchNormalization(axis = 3, scale = False)(resid5)
x = layers.LeakyReLU(.3)(resid5)
x = layers.Conv2D(20,kernel_size = (3,3),padding = 'same',strides = 1)(x)
x = layers.BatchNormalization(axis = 3, scale = False)(x)
x = layers.LeakyReLU(.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dense(10, activation = 'relu')(x)
x = layers.Dropout(.3)(x)
x = layers.LeakyReLU()(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)

experimentalModel1 = Model(inputs = inputs, outputs = outputs, name = 'experimental_model_1')
experimentalModel1.summary()
```

    Model: "experimental_model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_layer (InputLayer)        [(None, 128, 128, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d_218 (Conv2D)             (None, 128, 128, 20) 560         input_layer[0][0]                
    __________________________________________________________________________________________________
    batch_normalization_76 (BatchNo (None, 128, 128, 20) 60          conv2d_218[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu (LeakyReLU)         (None, 128, 128, 20) 0           batch_normalization_76[0][0]     
    __________________________________________________________________________________________________
    conv2d_219 (Conv2D)             (None, 64, 64, 20)   3620        leaky_re_lu[0][0]                
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, 64, 64, 20)   0           conv2d_219[0][0]                 
    __________________________________________________________________________________________________
    conv2d_220 (Conv2D)             (None, 64, 64, 20)   3620        leaky_re_lu_1[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_78 (BatchNo (None, 64, 64, 20)   60          conv2d_220[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, 64, 64, 20)   0           batch_normalization_78[0][0]     
    __________________________________________________________________________________________________
    conv2d_221 (Conv2D)             (None, 64, 64, 20)   3620        leaky_re_lu_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_79 (BatchNo (None, 64, 64, 20)   60          conv2d_221[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, 64, 64, 20)   0           batch_normalization_79[0][0]     
    __________________________________________________________________________________________________
    add (Add)                       (None, 64, 64, 20)   0           conv2d_219[0][0]                 
                                                                     leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    conv2d_222 (Conv2D)             (None, 32, 32, 20)   3620        add[0][0]                        
    __________________________________________________________________________________________________
    batch_normalization_80 (BatchNo (None, 32, 32, 20)   60          conv2d_222[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, 32, 32, 20)   0           batch_normalization_80[0][0]     
    __________________________________________________________________________________________________
    conv2d_223 (Conv2D)             (None, 32, 32, 20)   3620        leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_81 (BatchNo (None, 32, 32, 20)   60          conv2d_223[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)       (None, 32, 32, 20)   0           batch_normalization_81[0][0]     
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 32, 32, 20)   0           leaky_re_lu_5[0][0]              
                                                                     conv2d_222[0][0]                 
    __________________________________________________________________________________________________
    conv2d_224 (Conv2D)             (None, 16, 16, 20)   3620        add_1[0][0]                      
    __________________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)       (None, 16, 16, 20)   0           conv2d_224[0][0]                 
    __________________________________________________________________________________________________
    conv2d_225 (Conv2D)             (None, 16, 16, 20)   3620        leaky_re_lu_6[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_83 (BatchNo (None, 16, 16, 20)   60          conv2d_225[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)       (None, 16, 16, 20)   0           batch_normalization_83[0][0]     
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 16, 16, 20)   0           leaky_re_lu_7[0][0]              
                                                                     conv2d_224[0][0]                 
    __________________________________________________________________________________________________
    conv2d_226 (Conv2D)             (None, 8, 8, 20)     3620        add_2[0][0]                      
    __________________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)       (None, 8, 8, 20)     0           conv2d_226[0][0]                 
    __________________________________________________________________________________________________
    conv2d_227 (Conv2D)             (None, 8, 8, 20)     3620        leaky_re_lu_8[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_85 (BatchNo (None, 8, 8, 20)     60          conv2d_227[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)       (None, 8, 8, 20)     0           batch_normalization_85[0][0]     
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 8, 8, 20)     0           leaky_re_lu_9[0][0]              
                                                                     conv2d_226[0][0]                 
    __________________________________________________________________________________________________
    conv2d_228 (Conv2D)             (None, 4, 4, 20)     3620        add_3[0][0]                      
    __________________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)      (None, 4, 4, 20)     0           conv2d_228[0][0]                 
    __________________________________________________________________________________________________
    conv2d_229 (Conv2D)             (None, 4, 4, 20)     3620        leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_87 (BatchNo (None, 4, 4, 20)     60          conv2d_229[0][0]                 
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, 4, 4, 20)     0           batch_normalization_87[0][0]     
    __________________________________________________________________________________________________
    flatten_44 (Flatten)            (None, 320)          0           leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    dense_133 (Dense)               (None, 128)          41088       flatten_44[0][0]                 
    __________________________________________________________________________________________________
    dense_134 (Dense)               (None, 10)           1290        dense_133[0][0]                  
    __________________________________________________________________________________________________
    dropout_25 (Dropout)            (None, 10)           0           dense_134[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, 10)           0           dropout_25[0][0]                 
    __________________________________________________________________________________________________
    dense_135 (Dense)               (None, 1)            11          leaky_re_lu_12[0][0]             
    ==================================================================================================
    Total params: 83,249
    Trainable params: 82,929
    Non-trainable params: 320
    __________________________________________________________________________________________________
    


```python
experimentalModel1.compile(
    optimizer =  keras.optimizers.RMSprop(learning_rate=.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```


```python
experimental1Model_filepath = os.path.join('exmodel1')
experimental1ModelEarlyStop = EarlyStopping(patience=5, mode = 'auto', restore_best_weights=False)
experimental1ModelCheckpoint = ModelCheckpoint(experimental1Model_filepath,save_best_only=True)
```


```python
exHistory = experimentalModel1.fit(
    train,
    epochs=50,
    batch_size=32,
    validation_data=valImagesModel,
    class_weight = scale_weight(1),
    callbacks=[experimental1ModelCheckpoint, experimental1ModelEarlyStop]
)
```

    Epoch 1/50
    163/163 [==============================] - ETA: 0s - loss: 0.8645 - accuracy: 0.7987INFO:tensorflow:Assets written to: exmodel1/assets
    163/163 [==============================] - 26s 161ms/step - loss: 0.8645 - accuracy: 0.7987 - val_loss: 1.7544 - val_accuracy: 0.6290
    Epoch 2/50
    163/163 [==============================] - 21s 126ms/step - loss: 0.4898 - accuracy: 0.8907 - val_loss: 2.1108 - val_accuracy: 0.6290
    Epoch 3/50
    163/163 [==============================] - ETA: 0s - loss: 0.3651 - accuracy: 0.9197INFO:tensorflow:Assets written to: exmodel1/assets
    163/163 [==============================] - 26s 159ms/step - loss: 0.3651 - accuracy: 0.9197 - val_loss: 1.6140 - val_accuracy: 0.6290
    Epoch 4/50
    163/163 [==============================] - 20s 125ms/step - loss: 0.3130 - accuracy: 0.9189 - val_loss: 3.3740 - val_accuracy: 0.6290
    Epoch 5/50
    163/163 [==============================] - 20s 125ms/step - loss: 0.2815 - accuracy: 0.9191 - val_loss: 2.5323 - val_accuracy: 0.6290
    Epoch 6/50
    163/163 [==============================] - 20s 126ms/step - loss: 0.2735 - accuracy: 0.9258 - val_loss: 3.2565 - val_accuracy: 0.6290
    Epoch 7/50
    163/163 [==============================] - 21s 126ms/step - loss: 0.2468 - accuracy: 0.9308 - val_loss: 2.9589 - val_accuracy: 0.6290
    Epoch 8/50
    163/163 [==============================] - ETA: 0s - loss: 0.2358 - accuracy: 0.9279INFO:tensorflow:Assets written to: exmodel1/assets
    163/163 [==============================] - 26s 159ms/step - loss: 0.2358 - accuracy: 0.9279 - val_loss: 0.8384 - val_accuracy: 0.6290
    Epoch 9/50
    163/163 [==============================] - 21s 126ms/step - loss: 0.2230 - accuracy: 0.9363 - val_loss: 3.9983 - val_accuracy: 0.6290
    Epoch 10/50
    163/163 [==============================] - 21s 126ms/step - loss: 0.2167 - accuracy: 0.9346 - val_loss: 3.4998 - val_accuracy: 0.6290
    Epoch 11/50
    163/163 [==============================] - 20s 125ms/step - loss: 0.2103 - accuracy: 0.9331 - val_loss: 3.0585 - val_accuracy: 0.6290
    Epoch 12/50
    163/163 [==============================] - 20s 125ms/step - loss: 0.2195 - accuracy: 0.9277 - val_loss: 1.7393 - val_accuracy: 0.6290
    Epoch 13/50
    163/163 [==============================] - 20s 126ms/step - loss: 0.2069 - accuracy: 0.9410 - val_loss: 2.5964 - val_accuracy: 0.6290
    


```python
yex1_hat_train = experimentalModel1.predict(test).round()
yex1_train = test.y
confustion_matrix(yex1_train, yex1_hat_train)
```


```python
experimentalModel1.evaluate(testImagesModel)
experimentalModel1.evaluate(valImagesModel)
```

    16/16 [==============================] - 0s 8ms/step - loss: 2.2628 - accuracy: 0.6240
    4/4 [==============================] - 0s 8ms/step - loss: 2.5964 - accuracy: 0.6290
    




    [2.596433639526367, 0.6290322542190552]




```python
dfEx1Model = pd.DataFrame.from_dict(exHistory.history)
fig, ((ax1, ax2)) = plt.subplots(nrows = 1,ncols = 2, figsize = (14,7))
dfEx1Model.plot(y = ['loss', 'val_loss'],ax = ax1, title = 'Loss Metrics');
dfEx1Model.plot(y = ['accuracy', 'val_accuracy'],ax = ax2, title = 'Accuracy Metrics');
```


![png](output_71_0.png)



```python
experimentalModel1.evaluate(test)
```

## Multiple Inputs


```python
inputs = Input(shape = (128,128,1), name = 'input_layer')
x1 = layers.Conv2D(10,kernel_size = (15,15),padding = 'same', strides=1, activation = 'relu')(inputs)
x1 = layers.MaxPool2D(2, padding = 'same')(x1)
x1 = layers.Conv2D(10,kernel_size = (15,15),padding = 'same', strides=1, activation = 'relu')(x1)
x1 = layers.MaxPool2D(2, padding = 'same')(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(10, activation = 'relu')(x1)
x1 = layers.Dense(1, activation = 'sigmoid')(x1)

x2 = layers.Conv2D(10,kernel_size = (11,11),padding = 'same', strides=1, activation = 'relu')(inputs)
x2 = layers.MaxPool2D(2, padding = 'same')(x2)
x2 = layers.Conv2D(10,kernel_size = (11,11),padding = 'same', strides=1, activation = 'relu')(x2)
x2 = layers.MaxPool2D(2, padding = 'same')(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(10, activation = 'relu')(x2)
x2 = layers.Dense(1, activation = 'sigmoid')(x2)

x3 = layers.Conv2D(10,kernel_size = (7,7),padding = 'same', strides=1, activation = 'relu')(inputs)
x3 = layers.MaxPool2D(2, padding = 'same')(x3)
x3 = layers.Conv2D(10,kernel_size = (7,7),padding = 'same', strides=1, activation = 'relu')(x3)
x3 = layers.MaxPool2D(2, padding = 'same')(x3)
x3 = layers.Flatten()(x3)
x3 = layers.Dense(10, activation = 'relu')(x3)
x3 = layers.Dense(1, activation = 'sigmoid')(x3)

x4 = layers.Conv2D(10,kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(inputs)
x4 = layers.MaxPool2D(2, padding = 'same')(x4)
x4 = layers.Conv2D(10,kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(x4)
x4 = layers.MaxPool2D(2, padding = 'same')(x4)
x4 = layers.Flatten()(x4)
x4 = layers.Dense(10, activation = 'relu')(x4)
x4 = layers.Dense(1, activation = 'sigmoid')(x4)


block1 = layers.add([x1, x2, x3, x4])

# x = layers.MaxPool2D(2, padding = 'same')(block1)
# x = layers.Conv2D(10, kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(x)
# x = layers.MaxPool2D(2, padding = 'same')(x)
# x = layers.Flatten()(x)
x = layers.Dense(100, activation = 'relu')(block1)
x = layers.Dropout(.3)(x)
x = layers.Dense(10, activation - 'relu')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)

experimentalModel2 = Model(inputs = inputs, outputs = outputs, name = 'experimental_model_1')
experimentalModel2.summary()
```


```python
experimentalModel2.compile(
    optimizer =  'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```


```python
experimental2Model_filepath = os.path.join('exmodel2')
experimental2ModelEarlyStop = EarlyStopping(patience=5, mode = 'auto', restore_best_weights=False)
experimental2ModelCheckpoint = ModelCheckpoint(experimental2Model_filepath,save_best_only=True)
```


```python
exHistory = experimentalModel2.fit(
    alteredTrainImage,
    epochs=50,
    batch_size=32,
    validation_data=valImageModel,
    class_weight = {0:weights[0], 1:weights[1]},
    callbacks=[experimental2ModelCheckpoint, experimental2ModelEarlyStop]
)
```

## Experimental 3


```python
inputs = Input(shape = (128,128,1), name = 'input_layer')
x1 = layers.Conv2D(10,kernel_size = (15,15),padding = 'same', strides=1, activation = 'relu')(inputs)
x1 = layers.MaxPool2D(2, padding = 'same')(x1)

x2 = layers.Conv2D(10,kernel_size = (11,11),padding = 'same', strides=1, activation = 'relu')(inputs)
x2 = layers.MaxPool2D(2, padding = 'same')(x2)

x3 = layers.Conv2D(10,kernel_size = (7,7),padding = 'same', strides=1, activation = 'relu')(inputs)
x3 = layers.MaxPool2D(2, padding = 'same')(x3)

x4 = layers.Conv2D(10,kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(inputs)
x4 = layers.MaxPool2D(2, padding = 'same')(x4)

block1 = layers.add([x1, x2, x3, x4])

x1 = layers.Conv2D(10,kernel_size = (15,15),padding = 'same', strides=1, activation = 'relu')(block1)
x1 = layers.MaxPool2D(2, padding = 'same')(x1)


x2 = layers.Conv2D(10,kernel_size = (11,11),padding = 'same', strides=1, activation = 'relu')(block1)
x2 = layers.MaxPool2D(2, padding = 'same')(x2)


x3 = layers.Conv2D(10,kernel_size = (7,7),padding = 'same', strides=1, activation = 'relu')(block1)
x3 = layers.MaxPool2D(2, padding = 'same')(x3)


x4 = layers.Conv2D(10,kernel_size = (3,3),padding = 'same', strides=1, activation = 'relu')(block1)
x4 = layers.MaxPool2D(2, padding = 'same')(x4)

block2 = layers.add([x1, x2, x3, x4])
x = layers.Flatten()(block2)
x = layers.Dense(10, activation = 'relu')(x)
outputs = layers.Dense(1, activation = 'sigmoid')(x)

experimentalModel3 = Model(inputs = inputs, outputs = outputs, name = 'experimental_model_1')
experimentalModel3.summary()
```

    Model: "experimental_model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_layer (InputLayer)        [(None, 128, 128, 1) 0                                            
    __________________________________________________________________________________________________
    conv2d_200 (Conv2D)             (None, 128, 128, 10) 2260        input_layer[0][0]                
    __________________________________________________________________________________________________
    conv2d_201 (Conv2D)             (None, 128, 128, 10) 1220        input_layer[0][0]                
    __________________________________________________________________________________________________
    conv2d_202 (Conv2D)             (None, 128, 128, 10) 500         input_layer[0][0]                
    __________________________________________________________________________________________________
    conv2d_203 (Conv2D)             (None, 128, 128, 10) 100         input_layer[0][0]                
    __________________________________________________________________________________________________
    max_pooling2d_139 (MaxPooling2D (None, 64, 64, 10)   0           conv2d_200[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_140 (MaxPooling2D (None, 64, 64, 10)   0           conv2d_201[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_141 (MaxPooling2D (None, 64, 64, 10)   0           conv2d_202[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_142 (MaxPooling2D (None, 64, 64, 10)   0           conv2d_203[0][0]                 
    __________________________________________________________________________________________________
    add_29 (Add)                    (None, 64, 64, 10)   0           max_pooling2d_139[0][0]          
                                                                     max_pooling2d_140[0][0]          
                                                                     max_pooling2d_141[0][0]          
                                                                     max_pooling2d_142[0][0]          
    __________________________________________________________________________________________________
    conv2d_204 (Conv2D)             (None, 64, 64, 10)   22510       add_29[0][0]                     
    __________________________________________________________________________________________________
    conv2d_205 (Conv2D)             (None, 64, 64, 10)   12110       add_29[0][0]                     
    __________________________________________________________________________________________________
    conv2d_206 (Conv2D)             (None, 64, 64, 10)   4910        add_29[0][0]                     
    __________________________________________________________________________________________________
    conv2d_207 (Conv2D)             (None, 64, 64, 10)   910         add_29[0][0]                     
    __________________________________________________________________________________________________
    max_pooling2d_143 (MaxPooling2D (None, 32, 32, 10)   0           conv2d_204[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_144 (MaxPooling2D (None, 32, 32, 10)   0           conv2d_205[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_145 (MaxPooling2D (None, 32, 32, 10)   0           conv2d_206[0][0]                 
    __________________________________________________________________________________________________
    max_pooling2d_146 (MaxPooling2D (None, 32, 32, 10)   0           conv2d_207[0][0]                 
    __________________________________________________________________________________________________
    add_30 (Add)                    (None, 32, 32, 10)   0           max_pooling2d_143[0][0]          
                                                                     max_pooling2d_144[0][0]          
                                                                     max_pooling2d_145[0][0]          
                                                                     max_pooling2d_146[0][0]          
    __________________________________________________________________________________________________
    flatten_35 (Flatten)            (None, 10240)        0           add_30[0][0]                     
    __________________________________________________________________________________________________
    dense_80 (Dense)                (None, 10)           102410      flatten_35[0][0]                 
    __________________________________________________________________________________________________
    dense_81 (Dense)                (None, 1)            11          dense_80[0][0]                   
    ==================================================================================================
    Total params: 146,941
    Trainable params: 146,941
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
experimentalModel3.compile(
    optimizer =  'rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```


```python
experimental3Model_filepath = os.path.join('exmodel3')
experimental3ModelEarlyStop = EarlyStopping(patience=9, mode = 'auto', restore_best_weights=True)
experimental3ModelCheckpoint = ModelCheckpoint(experimental3Model_filepath,save_best_only=True)
```


```python
ex3History = experimentalModel3.fit(
    train,
    epochs=50,
    batch_size=32,
    validation_data=val,
    class_weight = {0:weights[0], 1:weights[1]},
    callbacks=[experimental3ModelCheckpoint, experimental3ModelEarlyStop]
)
```

    Epoch 1/50
    163/163 [==============================] - 241s 1s/step - loss: 0.8237 - accuracy: 0.6286 - val_loss: 0.7336 - val_accuracy: 0.5726
    Epoch 2/50
    163/163 [==============================] - 241s 1s/step - loss: 0.5895 - accuracy: 0.7412 - val_loss: 0.6435 - val_accuracy: 0.5806
    Epoch 3/50
    163/163 [==============================] - 243s 1s/step - loss: 0.5254 - accuracy: 0.7745 - val_loss: 0.6156 - val_accuracy: 0.7016
    Epoch 4/50
    163/163 [==============================] - 242s 1s/step - loss: 0.6814 - accuracy: 0.8016 - val_loss: 0.7182 - val_accuracy: 0.6210
    Epoch 5/50
    163/163 [==============================] - 244s 1s/step - loss: 0.4926 - accuracy: 0.7947 - val_loss: 0.4853 - val_accuracy: 0.7984
    Epoch 6/50
    163/163 [==============================] - 241s 1s/step - loss: 0.4907 - accuracy: 0.8083 - val_loss: 0.5871 - val_accuracy: 0.7177
    Epoch 7/50
    163/163 [==============================] - 241s 1s/step - loss: 0.4642 - accuracy: 0.8025 - val_loss: 0.5376 - val_accuracy: 0.7258
    Epoch 8/50
    163/163 [==============================] - 241s 1s/step - loss: 0.4468 - accuracy: 0.8167 - val_loss: 0.6788 - val_accuracy: 0.6532
    Epoch 9/50
    163/163 [==============================] - 241s 1s/step - loss: 0.4222 - accuracy: 0.8192 - val_loss: 0.5593 - val_accuracy: 0.7177
    Epoch 10/50
    163/163 [==============================] - 242s 1s/step - loss: 0.4538 - accuracy: 0.8209 - val_loss: 0.4784 - val_accuracy: 0.7661
    Epoch 11/50
    163/163 [==============================] - 242s 1s/step - loss: 0.4062 - accuracy: 0.8273 - val_loss: 0.6707 - val_accuracy: 0.6532
    Epoch 12/50
      6/163 [>.............................] - ETA: 3:24 - loss: 0.4867 - accuracy: 0.8073


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-114-dbfe5e045a62> in <module>
          5     validation_data=val,
          6     class_weight = {0:weights[0], 1:weights[1]},
    ----> 7     callbacks=[experimental3ModelCheckpoint, experimental3ModelEarlyStop]
          8 )
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\keras\engine\training.py in _method_wrapper(self, *args, **kwargs)
        106   def _method_wrapper(self, *args, **kwargs):
        107     if not self._in_multi_worker_mode():  # pylint: disable=protected-access
    --> 108       return method(self, *args, **kwargs)
        109 
        110     # Running inside `run_distribute_coordinator` already.
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1096                 batch_size=batch_size):
       1097               callbacks.on_train_batch_begin(step)
    -> 1098               tmp_logs = train_function(iterator)
       1099               if data_handler.should_sync:
       1100                 context.async_wait()
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\def_function.py in __call__(self, *args, **kwds)
        778       else:
        779         compiler = "nonXla"
    --> 780         result = self._call(*args, **kwds)
        781 
        782       new_tracing_count = self._get_tracing_count()
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\def_function.py in _call(self, *args, **kwds)
        805       # In this case we have created variables on the first call, so we run the
        806       # defunned version which is guaranteed to never create variables.
    --> 807       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        808     elif self._stateful_fn is not None:
        809       # Release the lock early so that multiple threads can perform the call
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\function.py in __call__(self, *args, **kwargs)
       2827     with self._lock:
       2828       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 2829     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2830 
       2831   @property
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\function.py in _filtered_call(self, args, kwargs, cancellation_manager)
       1846                            resource_variable_ops.BaseResourceVariable))],
       1847         captured_inputs=self.captured_inputs,
    -> 1848         cancellation_manager=cancellation_manager)
       1849 
       1850   def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1922       # No tape is watching; skip to running the function.
       1923       return self._build_call_outputs(self._inference_function.call(
    -> 1924           ctx, args, cancellation_manager=cancellation_manager))
       1925     forward_backward = self._select_forward_and_backward_functions(
       1926         args,
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\function.py in call(self, ctx, args, cancellation_manager)
        548               inputs=args,
        549               attrs=attrs,
    --> 550               ctx=ctx)
        551         else:
        552           outputs = execute.execute_with_cancellation(
    

    ~\.conda\envs\learn-env\lib\site-packages\tensorflow\python\eager\execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         58     ctx.ensure_initialized()
         59     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
    ---> 60                                         inputs, attrs, num_outputs)
         61   except core._NotOkStatusException as e:
         62     if name is not None:
    

    KeyboardInterrupt: 


# Visualizing our model's activations
Neural networks are vague by their very nature. Begin at a random point and optimize a buch of weights until we've reached a local loss function minimum. Trying to explain this process for image recognition with an explanation of the underlying process is inherantly flawed. When you see an image of your friends, can you describe how your brain was able to translate a bunch of pixel data into an understanding of the person those pixels are depicting? The answer (unless you've recently made some impressive breakthroughs in neural science) is no. CNNs are much the same. A much more useful way to get an understanding of what's going on is to break images down into features that can intuitively be assembled together to create understanding. The following cells contain images which depict what the neural network was "seeing" at various parts of the learning process.

<img src = 'model-activations.png'>


```python
# finalModel = keras.models.load_model('drive/MyDrive/Colab Notebooks/flatiron school/second_model_94-05_recall_97/')
# finalModel.summary()
```


```python
# Extract model layer outputs
layer_outputs = [layer.output for layer in finalModel.layers[:2]]

# Rather then a model with a single output, we are going to make a model to display the feature maps
activation_model = keras.models.Model(inputs=finalModel.input, outputs=layer_outputs)
```


```python
activations = activation_model.predict(val[0][0])
# We slice the third channel and preview the results
fig, axes = plt.subplots(4,4, figsize = (16,16))
for i in range(16):
    row = i//4
    column = i%4
    ax = axes[row, column]
    first_layer_activation = activations[0]
    ax.matshow(first_layer_activation[0, :, :, i], cmap='viridis')
plt.show()
```

# Business Insights and Conclusions
### Was the Project a Success?
In short, yes!
### Recommendations
#### 1. Medical Triage:
With an apperant ability to classify patients with Pneumonia 97% of the time, the model can be an effective triage mechanism for both emergancy and non-emergancy cases. The model should be integrated into the medical imaging work flow and predictions should be rendered at the time patient gets the image taken. These predictions can be used to prioritize patients the model sees as having pneumonia by flagging their charts as likely positive cases. Combining this knowledge with individual patient health knowledge can be used to prioritize patients at higher risk if treatment isn't rendered quickly. 

#### 2. System Redundancy:
Predictive models can serve as a redundant safety and liability measure for radiologists. Radiologist preformace is not perfect. While mistakes happen for a variety of reasons, there is little in the way of arguing they are necessary for the industry. Having a model that predicts pneumonia with a high degree of accuracy offers an additional safety net for against mis-diagnoses as the probability that both the model and the radiologist are wrong is less likely than either of them getting a prediction wrong. This safety net provides two benefits to the industry. Patients are less likely to be caused harm due to a mis-diagnoses, a clear win. Also, as the world of radiology is plagued with malpractice suits for failiar to diagnose something on a diagnostic image, the model will offer an additional insurance against a mis-diagnosis. 

#### 3. Record Conflicting Predictions:
Neural Networks can be greatly improved with increasing edge case detection. Edge cases are the difficult example which are routinely predicted incorrectly across architectures. If this model is integrated into the medical workflow, there will be examples in which the radiologist and the model have opposite predictions. These cases should be saved and specially catalouged as they are a reasonable standard for an edge case. Knowning difficult examples can be a huge advantage to a neural network as models allow for specific tuning to give edge cases more weight. These cases can be re-trained on and will greatly increase the model's effectiveness at correctly predicting the hard cases benefitting both medical science - by perhaps giving insight in the why the classification was difficult - and patient health. 
    
## Project Summary
Success with a graduated task is often in the eyes of the beholder. Is the final model produced here the absolute best model in the world. No. Does it rank in the top 100 models? Also, probably no. Would the model be a genuine benefit to a workflow in the medical industry that currently lacks AI technology? Absolutely. The ability to capture 97% of pnumonia patients based on a chest x-ray is pretty incredible considering a study published in the National Center for Biotechnology Informations journal found in a study that among various groups of radiologists the best average achieved by any category was just under 86% [1]. Granted, the study involved a blind assessment of an x-ray for all possible conditions and not a binary classification. Still, despite the comparison not being entirely fair, the model does appear to be preforming admirably. 

If considering the model to be roughly as good as a radiologist at diagnosing pnuemonia specifically, then our model offers a competant second opinion in a faction of the time and - if configured correctly in software - available at the time of the x-ray it self and not days later. A decision by the model and a subsequent radiologist are certainly not independent events but it stands to reason that the probability of both predictors incorrectly identifiying a patients diagnosis is lower than each individually (probably close to 0). To summarize, the model offers significant benefits in terms of being able to diagnose patients sick with pnuenomia 97% of the time. Combine this an immediate classification triage and a second round check for the eventual radiologist review, and this model's worth would be measured in lives saved.


# Deployment
The best model in this project has been deployed as a useable endpoint at [https://mod4-dash.herokuapp.com/](https://)

# Future Work
## 1. Additional image augmentation and sampling techniquies
   The fourth model's performace was underwhelming. In general more data should almost always lead to better neural network performace. In the case of the fourth model, we observed a slight decrease in performace when compared to the second model. Further investigation is needed to determine why this was the case. Because we used augmented data there is a possibility the model is recognizing the augmented images as belonging to the originals data set. Changing more of the data augmentation parameters should tease out what is happening. 
    

## 2. Non-Sequential Modeling

   Keras has an additional AIP called the "functional API" which allows for non-sequntial architectures. This feature has many more possibilities for creating effective model architectures including short circuit paths for layer perssistnace as well as spliting and joining modeling pipelines. 

# References
1. Assessing the accuracy and certainty in interpreting chest X-rays in the medical division
I Satia, S Bashagha, A Bibi, R Ahmed, S Mellor, F Zaman
Clin Med (Lond) 2013 Aug; 13(4): 349–352. doi: 10.7861/clinmedicine.13-4-349
PMCID: PMC4954299

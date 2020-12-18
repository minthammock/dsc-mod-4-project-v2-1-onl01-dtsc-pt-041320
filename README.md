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

# Data Understanding
This project is all about x-rays. Loading in a couple different images we see the images are composed of various sizes and pixel brightnesses. There is existing class imbalance in this data of about 3:1 pneumonia to non-pneumonia. In total there are approximately 6000 images which we will be working with. Here are a couple Example images to get you oriented. The first image is the non-pneumonia case. The second is the pneumonia case. 


<img src = 'normal-image1.jpg'>
<img src = 'pneumonia-image-1.jpg'>


#  Model: Advanced Connvulution Model
## Overview
The second model will rely on convulutional and pooling layers. A true deep dive into the underlying workings of these layers is beyond the scope of this notebook but we will sum them up for convienence.
### Backgroud Information
#### Convulutional layers 
Convulutional layers are the life force of modern neural network image processing. Unlike dense layers which connect every node of a layer to every node of the next layer, convulutional layers only connect small portions of the input layers. This approach is modeled off the human ocular system and is a state of the art way for understanding high dimensional imformation that boils down to a smaller subset of features. Images are the connonical example of this, but there are also several other types of data that can be analyzed effectively in this way. Ultimately, the convulution layers serve as feature extractors. In almost all examples they are combined with traditiaonl dense layers which take the information filtered by the convulutional layers and create prediction based on the normal preceptron model.
#### Pooling Layers
Pooling layers are the balance to convulution layers. While not strictly necessary, pooling layers are often used to decrease the dimensionality of the data. Because dense are more or less required to lie on top of the convulution layers to preform the "thinking" part of the model, we run into a dimensionality problem when transitioning the data from the convulutional layers to the dense layers. Dense layers require 1-dimensional input where as convulutional layers are capable of analyzing and outputing multi-dimensional data (2-dimensions in this case). In order to create this transition we must flatten the 2-dimensional data into 1 dimension which explodes the number of required node connections for the first dense layer. A 256 X 256 pixel image to transition to a dense layer of 20 nodes requires 128,000 connections per filter used in the preceeding convulutional layer. This will quickly get out of hand. The solution is pooling layers. Pooling layers use summary statistics to combine areas of the convulutional layers whereby effectively reducing the output data's dimensionality while preserving most of the information from the inputs. While, reducing the number of features likely means a reduced accuracy rate, the output will still retain enough features to catagorize images with incredible accuracy (this shouldn't be too surprising if you think about how quickly you can recognize objects from your periferal vision.)
## Building the  model
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

## Model Evaluation
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


# Third Party Benchmark
So we've gone through the trouble of creating a model but what does this mean in the grand scheme of things. Is the model good enough to compete in the data science industry? Are there other existing models that can vastly out perform what we've created?

In this case we can take these open ended questions and get some reasonable answer to them. Keras and several other popular neural network modules offer some pre-trained models whose architectures led them to victory in recent global challenges. One such architecture specifically trained on images is the VGG19 network. At this point we are going to load in this original model and play around with it as a more robust comparison of how our model is preforming within the industry. 

This process has several built in options. The model can come with all the layers from start to finish including both convulutional and fully connected layers already trained. There's also options to take only the connvulutional section pre-trained or to import the model architecture without any pre-training whatsoever. In this case, we will take the connvulutional layers having already been pre-trained as re-training them will take too long without considerable computing reasorces. We will create our own dense layers to add ot this model which will taylor the connvulutional layers for our specific task of classifying x-rays. While it seems odd to take a pre-trained state of the art model and add our own layers to it, because the underlying convulutional models are trained on a hugh variety of images, we need a fully connected architecture that will take the vast knowledge contained in the convulutional layers and funnel it down in a way that maximized our classification task. Unfortuneately, there's currently not a fantastic general structure that succeeds at any given task so there will be some trial and error. 



![png](output_61_1.png)

![png](output_61_3.png)


# Visualizing our model's activations
Neural networks are vague by their very nature. Begin at a random point and optimize a buch of weights until we've reached a local loss function minimum. Trying to explain this process for image recognition with an explanation of the underlying process is inherantly flawed. When you see an image of your friends, can you describe how your brain was able to translate a bunch of pixel data into an understanding of the person those pixels are depicting? The answer (unless you've recently made some impressive breakthroughs in neural science) is no. CNNs are much the same. A much more useful way to get an understanding of what's going on is to break images down into features that can intuitively be assembled together to create understanding. The following cells contain images which depict what the neural network was "seeing" at various parts of the learning process.

<img src = 'model-activations.png'>

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

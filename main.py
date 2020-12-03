import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics as km 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
  
def create_model_visuals(model, trainGenerator, valGenerator, testGenerator, batch_size, epochs, class_weight, train = True,kwargs = {}):
  '''
  This function takes a keras model and three Imagedataobject instances that represent
  your training, validation and testing sets and will generate a sk-learn style
  confustion matrix. If train = True, then the model will be trained for the number
  of epochs mentioned. At the end of the training the function will spit out the testing 
  set evaluation and some visuals pretaining to the history generated from the keras model.

  Parameters:

    model - keras.Model or Keras.Sequential instance. This model should 
            be created and compiled prior to passing them to this function

    trainGenerator - An keras.Imagedatagenerator object of the training data. 
                     This object can include augmentations as well. 

    valGenerator - A keras.Imagedatagenerator object of the validation data.

    testGenerator - An keras.Imagedatagenerator object including the test data.

    batch_size - Batch size (int) to pass through to the .fit method for the 
                 keras.Model. It should match the batch_size specified in the 
                 imagedatagenerator objects.

    epochs - Epochs (int) to pass through to the .fit method for the keras.Model.

    class_weight - class_weight (python dictionary) to pass through to the .fit 
                   method for the keras.Model

    train - Boolean. If Ture the function will refit the model. Please note that 
            this function will not re-instantiate the model so if you don't intend
            to continue the learning process from the prior fit you MUST re-create
            the model.

    kwargs - Additional parameters to pass to the .fit method that aren't specifically
             listed as parameters. 

  Returns:

    modelHistory - The history of the keras .fit method

    params - A call to the locals() method which saves all parameters fed into the 
             function as a dictionary.
  '''
  # call to locals to save the parameters fed in
  params = locals()

  # we define a helper function to create the confusion matrix visual
  def confustion_matrix(y, y_hat, normalize = 'true'):

    # instantiate a matplotlib figure object
    fig, ax = plt.subplots(1,1,figsize = (7,6))

    # a call to sk-learns confusion matrix function
    matrix = skm.confusion_matrix(y, y_hat, normalize=normalize,)

    # visualize the confusion matrix
    sns.heatmap(matrix, cmap = 'Blues', annot=True, ax = ax)

    # basic label setting
    ax.set(
      title = 'Confustion Matrix',
      xlabel = 'Predicted Label',
      ylabel = 'True Label'
    )
  
  # fit the model when train == True
  if train == True:
    modelHistory = model.fit(
        trainGenerator,
        batch_size=batch_size,
        epochs=epochs,
        class_weight = class_weight,
        **kwargs)
    
    # call eval on the testing set
    model.evaluate(testGenerator)

    # save the model history as a dictionary
    dfModel = pd.DataFrame().from_dict(modelHistory.history)

    # instantiate another figure object to display the training details
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2,ncols = 2, figsize = (18,7))
    dfModel.plot(y = ['loss', 'val_loss'],ax = ax1, title = 'Loss Metrics');
    dfModel.plot(y = ['accuracy', 'val_accuracy'],ax = ax2, title = 'Accuracy');
    dfModel.plot(y = ['val_true_positives'],ax = ax3, title = 'Val True Positives');
    dfModel.plot(y = [ 'val_true_negatives'],ax = ax4, title = 'Val True Negatives');
  
  # if train == False return a Nonetype
  else:
    modelHistory = None
  
  # feed the predicted labels and true labels into the sk-learn classification report function
  y_test_hat = np.where(model.predict(testGenerator) > .5, 1,0).flatten()
  y_test = testGenerator.y

  # create the confustion matrix
  confustion_matrix(y_test, y_test_hat)

  #create a pandas datafram version of the classification report
  dfTest = pd.DataFrame.from_dict(skm.classification_report(y_test, y_test_hat, output_dict=True))
  display(dfTest)
  
 
 
 
  return modelHistory,params
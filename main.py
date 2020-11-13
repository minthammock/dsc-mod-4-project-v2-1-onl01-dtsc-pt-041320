import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics as km 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

class keras_model_wrapper:
  def __init__(self,Model):
  	self.model = model
  def summary(self):
    print(self.model.summary())
  
def build_model(modelLayers, model_name, export = 'sklearn', **kwargs):
  try:
    inputs = modelLayers[0]
    x = None
    for i in range(len(modelLayers[:-1])):
      if i == 0:
        x = modelLayers[i]
      else:
        x = modelLayers[i](x)
  except:
    print("""The model has not been constructed. There is likely an issue with your layers configuration. 
		  Please check and make sure you provided an instance of keras.layers.Input as the first layer in 
		  the layers list. Other common problems are placing layers that are incompatable with each other
		  or layers are not in the correct order.""")
  try:
    outputs = modelLayers[-1](x)
  except(ValueError):
    print("The layer you provided as the output layer is not configured correctly. Please see keras error for more infomation.")
    print(ValueError)
  model_1 = Model(
    inputs = inputs, 
    outputs = outputs, 
    name = model_name)
  sequentialWrapper = Sequential()
  sequentialWrapper.add(modelLayers[0])
  sequentialWrapper.add(model_1)
  sequentialWrapper.compile(**kwargs)

  return sequentialWrapper
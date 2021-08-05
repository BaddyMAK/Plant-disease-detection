import keras.backend as K
import keras
from keras import layers, activations
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array
from keras.utils.vis_utils import plot_model
from keras.engine.topology import Layer
def VGG16_MODEL(nfilter): 
    #VGG16 like model
    model = Sequential([
        #block1
        layers.Conv2D(nfilter,(3,3),padding="same",name="block1_conv1",input_shape=(64,64,3)),
        layers.Activation("relu"),
        layers.BatchNormalization(),#improving the speed, performance, and stability of my CNN ,
        #normalize the input layer by re-centering and re-scaling
        layers.Dropout(rate=0.2),   

        layers.Conv2D(nfilter,(3,3),padding="same",name="block1_conv2"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),
        layers.MaxPooling2D((2,2),strides=(2,2),name="block1_pool"),

        #block2
        layers.Conv2D(nfilter*2,(3,3),padding="same",name="block2_conv1"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),

        layers.Conv2D(nfilter*2,(3,3),padding="same",name="block2_conv2"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),
        layers.MaxPooling2D((2,2),strides=(2,2),name="block2_pool"),

        #block3
        layers.Conv2D(nfilter*2,(3,3),padding="same",name="block3_conv1"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),

        layers.Conv2D(nfilter*4,(3,3),padding="same",name="block3_conv2"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),

        layers.Conv2D(nfilter*4,(3,3),padding="same",name="block3_conv3"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),
        layers.MaxPooling2D((2,2),strides=(2,2),name="block3_pool"),
        #layers.Flatten(),
        layers.GlobalAveragePooling2D(),

        #inference layer
        layers.Dense(128,name="fc1"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate=0.2),

        layers.Dense(128,name="fc2"),
        layers.BatchNormalization(),
        layers.Activation("relu"),    
        layers.Dropout(rate=0.2),

        layers.Dense(4,name="prepredictions"), # the number of labels in my model 
        layers.Activation("softmax",name="predictions")])

    return model

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self, inputNum, outputNum, layerInfo='dmmd'):
        super(MyModel, self).__init__()
        self.Layers = self.layersInit(inputNum, outputNum, layerInfo, nodeNum=20)
        self.d1 = Dense(units=inputNum, activation='relu')
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(outputNum)

    def call(self, x):
        for i in self.Layers:
            x = i(x)
        return x

    def layersInit(self, inputNum, outputNum, layerInfo, nodeNum=20):
        layers = []
        for i in range(len(layerInfo)-1):
            if i == 0:
                layers.append(Dense(inputNum, activation='sigmoid'))
            else:
                layers.append(Dense(nodeNum, activation='sigmoid'))
        layers.append(Dense(outputNum))
        return layers


# Create an instance of the model
model = MyModel(inputNum=2, outputNum=1)


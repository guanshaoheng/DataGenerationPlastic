import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class TensorFlowModel(tf.keras.Model):
    def __init__(self, outputNum, layerInfo='dmmd', dyWeight=1.0):
        super(TensorFlowModel, self).__init__(name='')
        self.layerInfo = layerInfo
        self.Layers = self.layersInit(outputNum, nodeNum=20)
        self.lossCalculator = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dyWeight = dyWeight

    @tf.function()
    def call(self, x, training=False):
        for i, l in enumerate(self.Layers[:-1]):
            if self.layerInfo[i] == 'd':
                x = l(x, training=training)
            else:
                x = l(x*x, training=training)
        x = self.Layers[-1](x, training=training)
        return x

    def layersInit(self, outputNum, nodeNum=40):
        layers = []
        for _ in self.layerInfo:
            layers.append(Dense(nodeNum, activation='relu'))
        layers.append(Dense(outputNum))
        return layers

    @tf.function()
    def gradient(self, x, training=False):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.call(x, training=training)
        dy = tape.batch_jacobian(y, x)
        return y, dy

    @tf.function()
    def hessian(self, x, training=False):
        with tf.GradientTape() as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                y = self.call(x, training=training)
            dy = tape1.batch_jacobian(y, x)
        ddy = tape2.batch_jacobian(target=dy, source=x)
        return y, dy, ddy

    def train_step(self, data):
        x, temp = data
        y, dy = temp
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred = self.call(x, training=True)  # Forward pass
            dy_pred = tape.jacobian(y_pred, x)
            # Compute our own loss
            loss = self.lossCalculator(y, y_pred) + self.dyWeight*self.lossCalculator(dy, dy_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def lossFunction(self, x, y, dy, training):
        y_pred, dy_pred = self.gradient(x, training=training)
        return self.lossCalculator(y, y_pred)+self.dyWeight*self.lossCalculator(dy, dy_pred)

    def grad(self, x, y, dy, training=True):
        # Create here your gradient and optimizor
        with tf.GradientTape() as tape:
            loss_value = self.lossFunction(x, y, dy, training=training)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)


# Create an instance of the model
if __name__ == "__main__":
    model = TensorFlowModel(outputNum=2)
    # model.summary()
    x = tf.random.normal(shape=[5, 3])
    y = tf.random.uniform(
        shape=[5, 2], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
    dy = tf.random.uniform(
        shape=[5, 2, 3], minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
    _ = model(x)
    model.summary()

    model.compile(optimizer='Adam', loss='mse')

    model.fit(x=x, y=[y, dy], epochs=10000)

    y, dy = model.gradient(x)
    y, dy, ddy = model.hessian(x)
    print(dy)



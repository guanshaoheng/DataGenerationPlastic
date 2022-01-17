import tensorflow as tf
import matplotlib.pyplot as plt

'''
                 !!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!
------------------------------------------------------------------------
    Reference: https://www.tensorflow.org/guide/advanced_autodiff
'''


def imshow_zero_center(image, **kwargs):
    lim = tf.reduce_max(abs(image))
    plt.imshow(image, vmin=-lim, vmax=lim, cmap='seismic', **kwargs)
    plt.colorbar()


def plot_as_patches(j):
    # Reorder axes so the diagonals will each form a contiguous patch.
    j = tf.transpose(j, [1, 0, 3, 2])
    # Pad in between each patch.
    lim = tf.reduce_max(abs(j))
    j = tf.pad(j, [[0, 0], [1, 1], [0, 0], [1, 1]],
               constant_values=-lim)
    # Reshape to form a single image.
    s = j.shape
    j = tf.reshape(j, [s[0]*s[1], s[2]*s[3]])
    imshow_zero_center(j, extent=[-0.5, s[2] - 0.5, s[0] - 0.5, -0.5])


x = tf.random.normal([7, 5])
# tf.random.uniform(
#     shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None
# )

layer1 = tf.keras.layers.Dense(8, activation=tf.nn.elu)
layer2 = tf.keras.layers.Dense(6, activation=tf.nn.elu)

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    y = layer1(x)
    y = layer2(y)

y_shape = y.shape

j = tape.jacobian(y, x)
j_shape = j.shape
imshow_zero_center(j[:, 0, :, 0])
_ = plt.title('A (batch, batch) slice')
plt.show()

plot_as_patches(j)
_ = plt.title('All (batch, batch) slices are diagonal')
plt.show()

j_sum = tf.reduce_sum(j, axis=2)
print(j_sum.shape)
j_select = tf.einsum('bxby->bxy', j)
print(j_select.shape)

jb = tape.batch_jacobian(y, x)
print(jb.shape)


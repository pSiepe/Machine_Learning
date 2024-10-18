from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import Input, Model
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf

x_train = np.load('Xtrain2_b.npy')
y_train = np.load('Ytrain2_b.npy')

# The pictures are given as 1 dim arrays of lenth 2304. So first we convert the input data and output data into 2 dim arrays of dim 48 by 48.
# The value of each picture stands for the grey scale.

list_of_images_exp_output = []
for image in y_train:
    image_reshaped = image.reshape(48, 48)
    image_with_channel = np.expand_dims(image_reshaped, axis=-1)
    image_normalized = image_with_channel / 255.0
    list_of_images_exp_output.append(image_normalized)

list_of_images_input = []
for image in x_train:
    image_reshaped = image.reshape(48, 48)
    image_with_channel = np.expand_dims(image_reshaped, axis=-1)
    image_normalized = image_with_channel / 255.0
    list_of_images_input.append(image_normalized)

arr_x_train = np.array(list_of_images_input)
arr_y_train = np.array(list_of_images_exp_output)

X_train, X_test, y_train, y_test = train_test_split(

    arr_x_train, arr_y_train, test_size=0.2, random_state=42)


# We create functions for the different parts of the network to make use of the reusability when we want to create multiple blocks.

def encoder_block(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    s = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(
        x)  # <- output of the previous layer is input of the next layer
    p = MaxPooling2D(pool_size=(2, 2), padding='same')(s)
    return s, p  # p provides the input to the next encoder block and s provides the context/features to the symmetrically opposte decoder block


# Baseline layer is just a bunch on Convolutional Layers to extract high level features from the downsampled Image
def baseline_layer(filters, inputs):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=1, activation='relu')(x)
    return x


# Decoder Block
def decoder_block(filters, connections, inputs):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), padding='same', activation='relu', strides=2)(inputs)
    skip_connections = concatenate([x, connections],
                                   axis=-1)  # <- characteristic of U-Net: encoder and decoder tensors are fused to create a tensor
    # with double the amount of channels.
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(skip_connections)
    x = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(x)
    return x


# Now we can begin building up the model. As a basis for orientation we will use the architecture described in: https://arxiv.org/abs/1505.04597

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    s_1, p_1 = encoder_block(64, inputs)
    s_2, p_2 = encoder_block(128, p_1)
    s_3, p_3 = encoder_block(256, p_2)

    x_1 = baseline_layer(512, p_3)

    x_2 = decoder_block(256, s_3, x_1)
    x_3 = decoder_block(128, s_2, x_2)
    x_4 = decoder_block(64, s_1, x_3)

    output = Conv2D(1, 1, activation='sigmoid')(x_4)

    model = Model(inputs=inputs, outputs=output, name='U-Net')

    return model


model = make_model(input_shape=(48, 48) + (1,))


# DICE Score = (2 * Intersection) / (Area of Set A + Area of Set B)
# “Intersection” refers to the number of overlapping or common elements (pixels or regions)
# between the predicted segmentation (Set A) and the ground truth segmentation (Set B).
# For evaluation of our models performance we will use the DICE metric.
# “Area of Set A” represents the total number of elements (pixels or regions) in the predicted segmentation.
# “Area of Set B” represents the total number of elements (pixels or regions) in the ground truth segmentation.
# In our case we have two binary matrices. By multiplying them only the values where they are both
# equal to one remain, hence the intersection of them.


def dice_score(exp_output, prediction):
    prediction_cl = tf.identity(prediction)  # <- tensorflow returns a tensor

    # Reshape prediction_cl to match exp_output
    prediction_cl = tf.reshape(prediction_cl,
                               tf.shape(exp_output))  # <- Reshape prediction_cl, as it is returned as a 1 dim array

    intersect = tf.reduce_sum(
        tf.multiply(prediction_cl, exp_output))  # <- calculate intersection using element-wise multiplication

    union = tf.reduce_sum(prediction_cl) + tf.reduce_sum(exp_output)
    dice = (2 * intersect + 1e-7) / (union + 1e-7)  # <- add small epsilon to avoid division by zero and cast to float32
    return dice


# 'loss=' will be used for the training and 'metrics=' for the evaluation of the model's performance

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),  # <- we already applied sigmoid at the end.
    metrics=[dice_score],
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    early_stopping
]

model.fit(
    x=X_train,
    y=y_train,
    batch_size=30,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=(X_test, y_test)
)



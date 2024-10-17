import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras.callbacks import EarlyStopping


x_train1 = np.load('Xtrain1.npy')
y_train1 = np.load('Ytrain1.npy')
print(len(x_train1))
# We know our images have size 48 times 48. They are given as a 1 dim array of length 2304. Overall there are 2783 images stored.
# Thus we have to reshape them into 2 dim format, to feed them into the CNN:

list_of_images = []
for image in x_train1:
    image_reshaped = image.reshape(48, 48)
    image_with_channel = np.expand_dims(image_reshaped, axis=-1)
    image_normalized = image_with_channel / 255.0
    list_of_images.append(image_normalized)

arr_x_train = np.array(list_of_images)

X_train, X_test, y_train, y_test = train_test_split(arr_x_train, y_train1, test_size=0.2, random_state=42)

#print(shape(X_train))

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=(48,48) + (1,), num_classes=2)
# Here we print out the model's structure, so the Layers and their order.
#keras.utils.plot_model(model, show_shapes=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    early_stopping
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    x = X_train,
    y = y_train,
    epochs=epochs,
    callbacks=callbacks,
    validation_data= (X_test, y_test)
)


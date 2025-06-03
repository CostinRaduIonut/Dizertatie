import keras.activations
from keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten, Rescaling, Dropout, BatchNormalization, Activation
from keras.models import Sequential, Model
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import keras

# sterge


# incarcam seturile de date pentru antrenare si predictii
gdrive = "/detection/recognition_dataset/"

image_shape = (28, 28, 1)
image_size = (image_shape[0], image_shape[1])
BATCH_SIZE = 16

data_training = image_dataset_from_directory(
    gdrive + "training_data",
    image_size = image_size,
    batch_size = BATCH_SIZE,
    label_mode = "int"
)



validation_data = image_dataset_from_directory(
    gdrive + "validation_data",
    image_size = image_size,
    batch_size = BATCH_SIZE,
    label_mode = "int"
)




# construim un model de retea neuronala convolutionala
def get_model():
    intrari = Input(image_shape)
    x = Rescaling(1./ 255)(intrari)
    x = Conv2D(filters=32, kernel_size=3)(intrari)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(filters=64, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(filters=256, kernel_size=3)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(.5)(x)

    iesiri = Dense(36, activation="softmax")(x)
    model = Model(inputs=intrari, outputs=iesiri)

    return model


model = get_model()

model.summary()
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],

)



rezultat = model.fit(
    data_training,
    epochs = 12,
    validation_data = validation_data
    
)

model.save("nn_braille")

acuratete = rezultat.history["accuracy"]
val_acuratete = rezultat.history["val_accuracy"]
cost = rezultat.history["loss"]
val_cost = rezultat.history["val_loss"]
epoci = range(1, len(acuratete) + 1)

plt.plot(epoci, acuratete, "bo", label = "Acuratetea antrenarii")
plt.plot(epoci, val_acuratete, "b", label = "Acuratetea validarii")
plt.title("Acuratetea antrenarii si validarii")
plt.legend()
plt.figure()

plt.plot(epoci, cost, "bo", label = "Costul antrenarii")
plt.plot(epoci, val_cost, "b", label = "Costul validarii")
plt.title("Costul antrenarii si validarii")
plt.legend()
plt.show()
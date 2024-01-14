import keras.optimizers
from keras import Sequential, applications, Model
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.preprocessing.image import ImageDataGenerator
from keras.optimizers import *

from graph import *

rescale_factor = 1./255
image_size = 125
batch_size = 35

train_data_dir = '/Users/cristobal/Desktop/archive4/raw-img/training'

train_datagen = ImageDataGenerator(
    rescale = rescale_factor,  # Normalizar los valores de los píxeles
    shear_range = 0.2,  # Rango para las transformaciones aleatorias
    zoom_range = 0.2,  # Rango para el zoom aleatorio
    horizontal_flip = True,  # Activar el giro horizontal aleatorio
    validation_split = 0.2)  # Establecer el porcentaje de imágenes para el conjunto de validación)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # Directorio con datos
    target_size = (125, 125),  # Cambiar el tamaño de las imágenes a 50x50
    batch_size = batch_size,
    shuffle = True,
    class_mode = 'categorical',  # 'binary' para clasificación binaria, 'categorical' para multiclase
    subset = 'training')  # Seleccionar solo el conjunto de entrenamiento

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,  # Directorio con datos
    target_size = (125, 125),  # Cambiar el tamaño de las imágenes a 50x50
    batch_size = batch_size,
    shuffle = True,
    class_mode = 'categorical',  # 'binary' para clasificación binaria, 'categorical' para multiclase
    subset = 'validation')  # Seleccionar solo el conjunto de entrenamiento

for imagen, etiqueta in train_generator:
    for i in range(10):

        plt.subplot(2, 5, i+1)
        plt.xticks()
        plt.yticks()
        plt.imshow(imagen[i])

    break

plt.show()

#Modelo

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(125, 125, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer='nadam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()


epochs = 30

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3, restore_best_weights=True)

history_of_train = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[es]
)

generatePlot(history_of_train)
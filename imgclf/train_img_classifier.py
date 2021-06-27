# from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def initialize_model(classes):
    resnet = ResNet50(input_shape=(224, 224, 3),
                      weights='imagenet', include_top=False)

    for layer in resnet.layers:
        layer.trainable = False

    x = Flatten()(resnet.output)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(inputs=resnet.input, outputs=prediction)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def get_dataset(path):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    training_set = train_datagen.flow_from_directory(path,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     subset="training", seed=42)

    test_set = test_datagen.flow_from_directory(path,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical',
                                                subset="validation", seed=42)
    return training_set, test_set


def train(epochs, path, classes):
    train, valid = get_dataset(path)
    model = initialize_model(classes)
    r = model.fit(
        train,
        validation_data=valid,
        epochs=int(epochs),
        steps_per_epoch=len(train),
        validation_steps=len(valid)
    )

    filename = f'{path}/score.h5'

    pickle.dump([round(r.history['loss'][-1], 4), round(r.history['accuracy'][-1]*100, 2), round(
        r.history['val_loss'][-1], 4), round(r.history['val_accuracy'][-1]*100, 2)], open(filename, 'wb'))

    model.save(f'{path}/model_resnet50.h5')


def predict(model_path, data_path):
    data_path = data_path + '/testFiles/'
    model = load_model(f'{model_path}/model_resnet50.h5')

    class_names = [dir for dir in os.listdir(
        model_path) if (dir[-3:] != '.h5')]
    images_list = os.listdir(data_path)

    loaded_images = [image.load_img(
        f'{data_path}/{img}', target_size=(224, 224)) for img in images_list]

    result = []
    for index, x in enumerate(loaded_images):
        x = image.img_to_array(x)
        x = x/255
        x = np.expand_dims(x, axis=0)
        # x=preprocess_input(x)
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        result.append([images_list[index], class_names[y_pred[0]]])

    return result

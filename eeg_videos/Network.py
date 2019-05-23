from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from eeg_videos.DataGenerator import NpyDataGenerator
from eeg_videos.DataPreprocessing import normalize


def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (2, 2), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(Conv2D(8, (2, 2), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    try:
        plot_model(model, to_file='model.jpg', show_layer_names=True, show_shapes=True)
    except ImportError:
        print('Unable to plot the model architecture - have you got installed pydot?\nSkipping plotting.\n')
    except OSError:
        print('Unable to plot the model architecture - have you got installed Graphviz?\nSkipping plotting.\n')

    return model


def train(model, directory, batch_size, input_shape, validation_split):
    train_data = NpyDataGenerator().flow_from_dir(directory=directory,
                                                  batch_size=batch_size,
                                                  shape=input_shape,
                                                  preprocessing_function=normalize,
                                                  validation_split=validation_split,
                                                  training_set=True)

    val_data = NpyDataGenerator().flow_from_dir(directory=directory,
                                                shape=input_shape,
                                                preprocessing_function=normalize,
                                                validation_split=validation_split,
                                                training_set=False)

    checkpoint = ModelCheckpoint('best_model', monitor='val_loss', save_best_only=True)

    model.fit_generator(train_data,
                        steps_per_epoch=64,
                        epochs=15,
                        validation_steps=2,
                        callbacks=[checkpoint],
                        validation_data=val_data)


def classify(input_shape, test_dir):
    test_data = NpyDataGenerator().flow_from_dir(directory=test_dir,
                                                 shape=input_shape,
                                                 preprocessing_function=normalize,
                                                 batch_size=8,
                                                 shuffle=False)

    model = load_model('best_model')

    output = model.predict_generator(test_data, steps=len(test_data), verbose=1)

    print(output)

    raise NotImplementedError('Finish me :)')

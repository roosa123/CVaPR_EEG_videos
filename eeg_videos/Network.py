import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from eeg_videos.DataGenerator import NpyDataGenerator
from eeg_videos.DataPreprocessing import normalize, kfold_data_sets, split_data
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from matplotlib import rcParams


def build_model(input_shape):
    """"
    This function builds the model and returns it - for now, it is idle, empty, stupid and totally uneducated.
    :param input_shape: the shape of the samples
    """
    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))       # adding convolutional layers...
    model.add(Conv2D(32, (2, 2), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))                            # adding max pooling layers...
    model.add(Dropout(0.5))                                                         # adding dropout layers....

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(Conv2D(8, (2, 2), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))

    model.add(Flatten())                        # flattening the data, so it can be passed to fully connected classifier

    model.add(Dense(512, activation='relu'))    # adding fully connected layers...
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])         # compile the model

    try:
        plot_model(model, to_file='model.jpg', show_layer_names=True, show_shapes=True)
    except ImportError:
        print('Unable to plot the model architecture - have you got installed pydot?\nSkipping plotting.\n')
    except OSError:
        print('Unable to plot the model architecture - have you got installed Graphviz?\nSkipping plotting.\n')

    return model


def train(model, directory, batch_size, input_shape, validation_split, method, k=200, train_split=0.9):

    """
    This function prepares the data generators (cool stuff, which will load the data from the
    hard drive dynamically, during training) and then - the coolest thing - trains the network.
    The coolest training can be maintained using Kfold method on prepared data-sets or simple one using one data set
    separated from
    :param model: model
    :param directory: the directory from which data should be loaded
    :param batch_size: size of the batch of samples
    :param input_shape: shape of the samples
    :param validation_split: amount of the data, which should be treated as validation set. Should be in range [0, 1]
    :param method: specifying method of maintaining input data and training (Kfold, simple)
    :param k: (optional, default 200) parameter k for Kfold method
    :param train_split: (optional, default 0.9) percentage of data used for training
    :return function returns test directory/ies for prosecuted training

    """

    if method == 'kfold':

        final_directory = '..\\DEAP\\sets_kfold'
        result = kfold_data_sets(directory=directory,
                                 final_directory=final_directory,
                                 k=k,
                                 method='random_var')

        if result == 1:
            print('Error while computing sets')
            return 1
        else:
            directories = result[0]
            validation_directories = result[1]

        print(directories)

        for i, directory_set in enumerate(directories):

            print('Training iteration: ', i+1)
            train_data = NpyDataGenerator().flow_from_dir(directory=directory_set,
                                                          batch_size=batch_size,
                                                          shape=input_shape,
                                                          preprocessing_function=normalize,
                                                          validation_split=0.0,
                                                          training_set=True)

            val_data = NpyDataGenerator().flow_from_dir(directory=validation_directories[i],
                                                        shape=input_shape,
                                                        preprocessing_function=normalize,
                                                        validation_split=1.0,
                                                        training_set=False)

            checkpoint = ModelCheckpoint('best_model', monitor='val_loss', save_best_only=True)

            model.fit_generator(train_data,
                                steps_per_epoch=64,
                                epochs=5,
                                validation_steps=2,
                                callbacks=[checkpoint],
                                validation_data=val_data)

        return 0

    elif method == 'simple':

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

        return 0

    else:

        print('Wrong method')
        return 1


def classify(input_shape, test_dir):

    """
    This functions attempts to classify samples from the test set.
    And then creates confusion matrix
    :param input_shape:
    :param test_dir:
    :return:
    """
    test_data = NpyDataGenerator().flow_from_dir(directory=test_dir,
                                                 shape=input_shape,
                                                 preprocessing_function=normalize,
                                                 batch_size=8,
                                                 shuffle=False)

    model = load_model('best_model')

    output = model.predict_generator(test_data, steps=len(test_data), verbose=1)

    rcParams['toolbar'] = 'None'

    predicted_classes = np.argmax(output, axis=1)
    true_classes = test_data.classes

    labels = ["Meditation", "Boredom", "Excitement", "Frustration"]

    pred = []
    true = []

    for i in range(len(predicted_classes)):
        pred.append(labels[predicted_classes[i]])
        true.append(labels[true_classes[i]])

    conf_matrix = confusion_matrix(true, pred, labels)

    width, height = conf_matrix.shape
    annots = np.empty_like(conf_matrix).astype(str)

    sums = np.sum(conf_matrix, axis=1, keepdims=True)
    percentage = conf_matrix / sums * 100

    for i in range(width):
        for j in range(height):
            if i == j:#'%.1f%%\n%d/%d' % (p, c, s)
                annots[i, j] = ("%.1f%%\n%d/%d" % (percentage[i, j], conf_matrix[i, j], sums[i]))
            else:
                annots[i, j] = ("%.1f%%\n%d" % (percentage[i, j], conf_matrix[i, j]))

    conf_matrix = DataFrame(conf_matrix)

    fig, ax = plt.subplots(figsize=(30, 15))
    plt.tight_layout(pad=2.5, h_pad=2.5, w_pad=2.5, rect=[0.2, 0.3, 1, 1])

    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix, annot=annots, fmt='', ax=ax)

    ax.set_xlabel("Predicted", fontsize=30)
    ax.set_ylabel("Actual", fontsize=30)

    ax.set_xticklabels(labels, rotation=45, fontsize=15)
    ax.set_yticklabels(labels, rotation=45, fontsize=15)

    plt.savefig('confusion_matrix.jpg')

    print(output)
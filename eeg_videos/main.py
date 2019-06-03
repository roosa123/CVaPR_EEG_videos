from eeg_videos.DataPreprocessing import load_data, preprocess_data, split_data
from eeg_videos.Network import build_model, train, classify

if __name__ == '__main__':
    data, labels, files = load_data()
    preprocess_data(data,
                    labels,
                    files,
                    directory='..\\DEAP\\train\\')
    print('Data has been successfully prepocessed and saved. Attempting to split it into training and test set...')
    train_split = 0.7
    train_set, test_set = split_data(directory='..\\DEAP\\train',
                                     final_directory='..\\DEAP\\SETS',
                                     train_split=train_split)
    print('Data has been successfully split into two sets. Attempting to train build the model...')
    in_shape = (120, 120, 4)
    network = build_model(input_shape=in_shape)
    print('The model has been successfully built. Attempting to train ti...')
    train(network,
          directory=train_set,
          batch_size=8,
          input_shape=in_shape,
          validation_split=0.3,
          method='kfold')
    print('The model has been successfully trained. Attempting to classify some cool samples from the test set...')
    classify(input_shape=in_shape, test_dir=test_set)


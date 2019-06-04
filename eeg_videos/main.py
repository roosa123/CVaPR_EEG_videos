from eeg_videos.DataPreprocessing import load_data, preprocess_data, split_data
from eeg_videos.Network import build_model, train, classify

if __name__ == '__main__':
    print('Here we go!')
    '''
    data, labels, files = load_data()
    preprocess_data(data,
                    labels,
                    files,
                    directory='..\\DEAP\\train\\')
    print('Data has been successfully prepocessed and saved. Attempting to split it into training and test set...')
    '''
    train_split = 0.7  # Should be between 0 and 1
    result = split_data(directory='..\\DEAP\\train_zp',
                        final_directory='..\\DEAP\\main_sets',
                        train_split=train_split)
    if result == 1:
        print('Error occured')
    else:
        train_set_dir = result[0]
        test_set_dir = result[1]

        print('Data has been successfully split into two sets. Attempting to train build the model...')
        print('Sets located in: ', train_set_dir, test_set_dir)
        in_shape = (120, 120, 4)
        network = build_model(input_shape=in_shape)
        print('The model has been successfully built. Attempting to train ti...')

        error = train(network,
                      directory=train_set_dir,
                      batch_size=8,
                      input_shape=in_shape,
                      validation_split=0.3,
                      method='kfold',
                      k=6)
        if not error:
            print('The model has been successfully trained. Attempting to classify some cool samples from the test set...')
            classify(input_shape=in_shape, test_dir=test_set_dir)
        else:
            print('Error occured')

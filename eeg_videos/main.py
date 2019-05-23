from .DataPreprocessing import load_data, preprocess_data
from .Network import build_model, train, classify

if __name__ == '__main__':
    data, labels, files = load_data()
    preprocess_data(data, labels, files, directory='..\\DEAP\\train\\')
    in_shape = (120, 120, 4)
    network = build_model(input_shape=in_shape)
    train(network, directory='..\\DEAP\\img', batch_size=8, input_shape=in_shape, validation_split=0.3)
    classify()

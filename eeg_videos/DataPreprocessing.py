from __future__ import division
import numpy as np
import os
import _pickle as cPickle
import matplotlib.pyplot as plt
# from scipy.stats import frechet_r_gen
import random
from shutil import copy

import EEG.EEG.eeg as eeg
from mne.time_frequency import psd_array_welch
from scipy.integrate import simps
from scipy.interpolate import griddata


def normalize(sample):
    """"
    This function is responsible for scaling the values in the input data to the range [-1, 1].
    Data normalization is generally essential when training the network -
    lack of it can cause troubles or even inability of the network to learn.

    The formula was derived from the widely known formula for scaling data to the range [a, b]:
    x = (b - a) * (x - min(x)) / (max(x) - min(x)) + a
    """
    return 2 * (np.divide(np.subtract(sample, np.min(sample)), (np.max(sample) - np.min(sample)))) - 1


def load_data():
    """"
    This function is responsible for loading the dataset from the disc.
    The path from which he dataset is obtained is constant.
    The function also assigns labels for each loaded sample.
    The labels are obtained on the basis of the valence and arousal values contained in the dataset.
    They are assigned according to the following rules:
        - 0 - Meditation:  Val > 5 Arousal < 5
        - 1 - Boredom:     Val < 5 Arousal < 5
        - 2 - Excitement:  Val > 5 Arousal > 5
        - 3 - Frustration: Val < 5 Arousal > 5
    """
    path = '../DEAP/data_preprocessed_python'
    list_of_labels = []
    list_of_data = []
    files = []

    for (_, _, sets) in os.walk(path):
        files.extend(sets)

    for file in files:

        participant_base = cPickle.load(open(path + '/' + file, 'rb'), encoding='latin1')
        # ----------Labels - four discrete  states---------------

        labels = participant_base['labels']
        n, _ = labels.shape
        labels2 = np.zeros([n, 2])

        # States - labels IMPORTANT
        # 0 - Meditation  Val > 5 Arousal < 5
        # 1 - Boredom     Val < 5 Arousal < 5
        # 2 - Excitement  Val > 5 Arousal > 5
        # 3 - Frustration Val < 5 Arousal > 5

        for i in range(1, n + 1):
            labels2[i - 1, 0] = i
            curr_valence = labels[i - 1, 0]
            curr_arousal = labels[i - 1, 1]
            if curr_valence > 5:
                if curr_arousal <= 5:
                    labels2[i - 1, 1] = 0
                else:
                    labels2[i - 1, 1] = 2
            else:
                if curr_arousal <= 5:
                    labels2[i - 1, 1] = 1
                else:
                    labels2[i - 1, 1] = 3

        # -------------- Lists for labels and data------------------

        data = participant_base['data']
        list_of_labels.append(labels2)
        list_of_data.append(np.delete(data, np.s_[32:], axis=1))
        # ----------------------------------------------------------

    print(len(list_of_labels))
    print(len(list_of_data), len(list_of_data[0]), len(list_of_data[0][0]))

    return list_of_labels, list_of_data, files


def preprocess_data(list_of_labels, list_of_data, files, directory):
    """"
    Data preprocessing. To obtain data in form, which can be easily processed by the neural network,
    the average band power in each channel is mapped into a spatial map (as suggested in Li, Y., Huang, J., Zhou, H.,
    & Zhong, N. (2017). Human emotion recognition with electroencephalographic multidimensional features by
    hybrid deep neural networks. Applied Sciences, 7(10), 1060. ). As a result, each trial is represented by four maps
    presenting spatial distribution of average band power in each of the four bands: alpha, beta, gamma and theta.
    After initial preparation of the data, power spectral density for each channel in each trial completed by
    each participant is computed. As long as we are not interested in processing each of the frequencies separately,
    the average band power is calculated for alpha, beta, gamma and theta bands (as suggested in Al-Nafjan, A., Hosny,
    M., Al-Wabil, A., & Al-Ohali, Y. (2017). Classification of human emotions from electroencephalogram (EEG) signal
    using deep neural network. International Journal of Advanced Computer Science and Applications, 8(9). ). The values
    computed this way are then mapped into a 9×9 matrix (the dimensions are maximum point numbers between
    the horizontal or vertical test points). The empty cells of the matrix are filled with average values stored
    in the surrounding cells, using formula from Li, Y., Huang, J., Zhou, H., & Zhong, N. (2017).
    Human emotion recognition with electroencephalographic multidimensional features by hybrid deep neural networks.
    Applied Sciences, 7(10), 1060. After the feature matrix is filled, it is used as a base table to generate
    data matrix through the interpolation method. The final data matrix has dimensions of 4×120×120 (corresponding
    to four bands and dimensions of the matrix of average band power obtained through the interpolation).
    """
    #  Next step is to extract features from all packages of Data

    # ============== some useful variables ==============

    # positions of the electrodes in tne NxN matrix - which will be then extended to the spatial feature map
    channel_positions = np.array([[0, 3], [1, 3], [2, 2], [2, 0], [3, 1], [3, 3], [4, 2], [4, 0],
                                  [5, 1], [5, 3], [6, 2], [6, 0], [7, 3], [8, 3], [8, 4], [6, 4],
                                  [0, 5], [1, 5], [2, 4], [2, 6], [2, 8], [3, 7], [3, 5], [4, 4],
                                  [4, 6], [4, 8], [5, 7], [5, 5], [6, 6], [6, 8], [7, 5], [8, 5]])

    # Coordinates of the points in the matrix, which contain data from electrodes or estimated on the basis of
    # the data from electrodes. The whole matrix contains useful data - not only cells, which contain
    # data from electrodes!
    # The coordinates will be then used in the interpolation
    points = np.zeros((81, 2))          # first, create array of zeros
    x1, y1 = 0, 0                       # initial values of coordinates will be (0, 0)

    for x in range(len(points)):        # fill the array of point coordinates
        points[x] = [x1, y1]            # so the resulting array will have form of:
        y1 += 1                         # [[0. 0.] [0. 1.] [0. 2.] ... [0. 8.]
        if y1 > 8:                      # [1. 0.] [1. 1.] [1. 2.] ... [1. 8.]
            y1 = 0                      # ...
            x1 += 1                     # [8. 0.] [8. 1.] [8. 2.] ... [8. 8.]]

    # ===================================================

    for i in range(len(list_of_data)):
        for j in range(len(list_of_data[i])):
            # prepare the array for the average band powers - we know, that we are supposed to have 9x9 matrix of
            # the real or estiamated data, and the average band power will be calculated in
            # 4 bands (alpha, beta, gamma, theta) - so the size of the array will be (9, 9, 4)
            avg_band_power = np.zeros((9, 9, 4))
            for k in range(len(list_of_data[i][j])):
                print('Processing sample %d, row %d, column %d...' % (i, j, k))
                data = list_of_data[i][j][k]

                # kod Marcela
                # ps = 10*np.log10(np.abs(np.fft.fft(data))**2)
                # time_step = 1 / 128
                # freqs = np.fft.fftfreq(data.size, time_step)
                # N = len(freqs)
                # freqs = np.abs(freqs[0:N//2])
                # idx = np.argsort(freqs)

                # kod znaleziony
                fs = 128                                    # sampling rate
                # compute FFT over the sample - should show peaks at some specific frequencies
                freqs, ps = eeg.computeFFT(data, fs=fs)
                idx = np.argsort(freqs)

                plt.plot(freqs[idx], ps[idx])
                plt.title(label=(files[i] + ' video: ' + str(j + 1) + ' chanel: ' + str(k + 1)))
                plt.show()

                # PSD
                # calculate PSD using Welch's method. Although better approach is to use multitaper algorithm,
                # but firstly, it is much slower than Welch's method. Secondly, our data is already
                # cleaned and preprocessed, so both methods probably will give similar results

                psds, freqs = psd_array_welch(data, sfreq=fs, n_per_seg=7, n_fft=np.shape(data)[0])

                plt.plot(freqs[1:np.shape(freqs)[0] - 1], psds[1:np.shape(psds)[0] - 1])
                plt.show()
                # 1. find average band powers (for alpha, beta, gamma and theta bands)
                freq_bands = {              # upper and lower limits of all the needed bands
                    'alpha': [8, 13],
                    'beta': [13, 30],
                    'gamma': [30, 40],
                    'theta': [4, 8]
                }

                freq_res = freqs[1] - freqs[0]      # frequency resolution
                avg_power = []                      # we have to store the avg band power somewhere...

                # calculate avg band power (absolute, not relative - we don't need percents) in each of the bands
                # The absolute delta power is equal to the area under the plot of already calculated PSD.
                # And as we know, this can be obtained by integrating. As we don't have any formula, which we can
                # integrate to obtain this area, we need to approximate it. As suggested in
                # https://raphaelvallat.com/bandpower.html, let's choose Simpson's method, which is realying on
                # decomposition of the area into several parabola and then summing up the area of these parabola.
                for band in ['alpha', 'beta', 'gamma', 'theta']:        # maybe strange, but definitely readable xD
                    # find indices, which satisfy the limits of the specific band
                    idx = np.logical_and(freqs >= freq_bands[band][0], freqs <= freq_bands[band][1])
                    avg = simps(psds[idx], dx=freq_res)                    # integrals, sweet integrals <3
                    avg_power.append(avg)                                  # success, save the computed value!

                # as long as we consider k-th channel, we should save the vector of
                # the already computed values of avg band powers into our previously prepared array
                # but in the correct position - one of those, which are supposed to contain data from electrodes,
                # already listed in channel_positions variable!
                avg_band_power[channel_positions[k][0]][channel_positions[k][1]] = avg_power

                print('\t\tAvg band pow for current done')

            # 2. calculate unknown points in the feature matrix
            # start from the middle, so the matrix filling won't ever start from a place surrounded by zeros
            vertical = False
            forward = True
            x, y = 4, 4
            max_x = x + 1
            min_x = x - 1
            max_y = y + 1
            min_y = y - 1
            matrix_len = 9

            cnt = 0

            #              __,aaPPPPPPPPaa,__
            #          ,adP"""'          `""Yb,_
            #       ,adP'                     `"Yb,
            #     ,dP'     ,aadPP"""""YYba,_     `"Y,
            #    ,P'    ,aP"'            `""Ya,     "Y,
            #   ,P'    aP'     _________     `"Ya    `Yb,
            #  ,P'    d"    ,adP""""""""Yba,    `Y,    "Y,
            # ,d'   ,d'   ,dP"            `Yb,   `Y,    `Y,
            # d'   ,d'   ,d'    ,dP""Yb,    `Y,   `Y,    `b
            # 8    d'    d'   ,d"      "b,   `Y,   `8,    Y,
            # 8    8     8    d'    _   `Y,   `8    `8    `b
            # 8    8     8    8     8    `8    8     8     8
            # 8    Y,    Y,   `b, ,aP     P    8    ,P     8
            # I,   `Y,   `Ya    """"     d'   ,P    d"    ,P
            # `Y,   `8,    `Ya         ,8"   ,P'   ,P'    d'
            #  `Y,   `Ya,    `Ya,,__,,d"'   ,P'   ,P"    ,P
            #   `Y,    `Ya,     `""""'     ,P'   ,d"    ,P'
            #    `Yb,    `"Ya,_          ,d"    ,P'    ,P'
            #      `Yb,      ""YbaaaaaadP"     ,P'    ,P'   Normand
            #        `Yba,                   ,d'    ,dP'    Veilleux
            #           `"Yba,__       __,adP"     dP"
            #               `"""""""""""""'
            #
            # in plain English: fill the empty cells in the matrix in the spiral order, beginning from the middle
            while cnt < matrix_len * matrix_len:
                if not np.any((channel_positions[:] == [y, x]).all(axis=1)):
                    for z in range(4):
                        k = 0
                        a = (avg_band_power[y + 1][x][z] if y + 1 < 9 else 0.0)
                        k += 0 if a == 0.0 else 1
                        b = (avg_band_power[y - 1][x][z] if y - 1 > -1 else 0.0)
                        k += 0 if a == 0.0 else 1
                        c = (avg_band_power[y][x + 1][z] if x + 1 < 9 else 0.0)
                        k += 0 if a == 0.0 else 1
                        d = (avg_band_power[y][x - 1][z] if x - 1 > -1 else 0.0)
                        k += 0 if a == 0.0 else 1

                        avg_band_power[y][x][z] = (a + b + c + d) / k if k != 0.0 else 0.0

                if not vertical:
                    if forward:
                        if x < max_x:
                            x += 1
                        else:
                            vertical = True
                            y += 1
                            max_x += 1
                    else:
                        if x > min_x:
                            x -= 1
                        else:
                            vertical = True
                            y -= 1
                            min_x -= 1
                else:
                    if forward:
                        if y < max_y:
                            y += 1
                        else:
                            vertical = False
                            forward = False
                            x -= 1
                            max_y += 1
                    else:
                        if y > min_y:
                            y -= 1
                        else:
                            vertical = False
                            forward = True
                            x += 1
                            min_y -= 1

                cnt += 1

            print('\t\t2 done')

            # now now, get ready for interpolation xD
            # first, we need the data to be stored in the 1D structure for single channel - so we can save the data
            # in the 2D array, from which we can then obtain the vector of values corresponding to the specific channel
            avg_bands = avg_band_power.reshape((81, 4))
            # coordinate vector in the X dimension - Xs' of the points, in which we should interpolate the values
            nx = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), num=120)
            # coordinate vector in the Y dimension - Ys' of the points, in which we should interpolate the values
            ny = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=120)
            # coordinate matrices from coordinate vectors
            grid_x, grid_y = np.meshgrid(nx, ny)

            out = []
            for band in range(4):
                # Wooohooo, here we go! Interpolate it!
                grid = griddata(points, avg_bands[:, band], (grid_x, grid_y), fill_value=0.0, method='cubic')
                # save the resulting matrix
                out.append(grid)

            # the data is supposed to be saved in the directory, which will indicate the class of the sample - ie.
            # the directory's name should be equal to the label assigned to the sample's class
            cur_dir = directory + str(list_of_labels[i][j][1])

            # if the directory already doesn't exist - make it
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            # sadly, our data is of shape (4, 120, 120). We cannot pass it to the network in this shape,
            # because convolutions and pooling will quickly downsample it below zero :(
            # so we should reshape it
            out = np.rollaxis(np.rollaxis(np.array(out), 2), 2)

            # ufff, done. Save the ready sample :)
            np.save(cur_dir + '\\' + str(i).zfill(2) + str(j).zfill(2) + '.npy', np.array(out))
            print('Sample successfully processed and saved into ' + cur_dir + '\\'
                  + str(i).zfill(2) + str(j).zfill(2) + '.npy')


def split_data(directory, final_directory, train_split):
    """

    Function for splitting data randomly into two sets - for training and testing.
    Function can be used as testing data generator.

    :param directory: path to files - should contain separated directories for each class
    :param final_directory: path for writing files - in which training and testing sets will occur
    :param train_split: (between 0-1) border in which data set should be splitted
    :return:
    """
    files = []
    dirs = []
    tab_nr = []
    nr_files = 0

    set_path = final_directory + "\\" + 'TRAIN'
    set_path_test = final_directory + "\\" + 'TEST'

    if not os.path.exists(directory):
        print('Not such a directory')
        return 1

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    for _, a, _ in os.walk(directory):
        dirs = a
        break

    for i, d in enumerate(dirs):
        tem_files = []
        for (_, _, tem_file) in os.walk(directory + '\\' + d):
            tem_files.extend(tem_file)
        for file in tem_files:
            files.append([i, file])
            # Saving label and filename
            nr_files += 1

    for i in range(nr_files):
        tab_nr.append(i)

    random.shuffle(tab_nr)

    train_amount = int(nr_files * train_split)

    for i in range(nr_files):
        file = files[tab_nr[i]]
        name = str(file[1])
        if i < train_amount:
            temp_path = set_path + "\\" + str(dirs[file[0]])
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            copy(directory + '\\' + str(dirs[file[0]]) + '\\' + name, temp_path)
        else:
            temp_path_test = set_path_test + "\\" + str(dirs[file[0]])
            if not os.path.exists(temp_path_test):
                os.makedirs(temp_path_test)
            copy(directory + '\\' + str(dirs[file[0]]) + '\\' + name, temp_path_test)

    print('Train set located in: ', set_path)
    return set_path, set_path_test


def kfold_data_sets(directory, final_directory, k,  method='random_var'):
    """
    Function for preparing iteration sets of data for using K-fold technique with
    random/straight choice.

    :param directory: path to files - should contain separated directories for each class
    :param final_directory: path for writing files - in which sets in separated folders will occur
    :param k: refers to number of training per iteration (all_samples - k -> training samples)
    :param  method: 'random'/ 'simple' training sets selection
    :return: list of directories to particular sets (for training and testing)
    """

    files = []
    dirs = []
    directories = []
    test_directories = []
    tab_nr = []
    nr_files = 0
    ch_point = 0

    if not os.path.exists(directory):
        print('Wrong directory')
        return 1

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    for _, a, _ in os.walk(directory):
        dirs = a
        break

    for i, d in enumerate(dirs):
        tem_files = []
        for (_, _, tem_file) in os.walk(directory + '\\' + d):
            tem_files.extend(tem_file)
        for file in tem_files:
            files.append([i, file])
            # Saving label and filename
            nr_files += 1

    if k > (0.2 * nr_files):
        print('K value should be lower than 20% of number of data in sets!')
        return 1

    for i in range(nr_files):
        tab_nr.append(i)

    if method == 'random_var':
        random.shuffle(tab_nr)
        tab_nr_shuffle = tab_nr
    elif method == 'simple':
        tab_nr_shuffle = tab_nr
    else:
        print('Wrong method')
        return 1

    mod = nr_files % k
    nr_sets = int(nr_files / k)

    for i in range(nr_sets):

        temp_files = files.copy()
        test_files = []
        set_path = final_directory + "\\" + 'ITER_' + str(i + 1)

        set_path_test = final_directory + "\\" + 'TEST_ITER' + str(i + 1)

        if not os.path.exists(set_path):
            os.makedirs(set_path)
        if not os.path.exists(set_path_test):
            os.makedirs(set_path_test)

        k_tem = k
        if i == 0 and mod != 0:
            k_tem = mod

        # Choosing test samples
        for j in range(ch_point, k_tem + ch_point):
            test_files.append(tab_nr_shuffle[j])
            temp_files[tab_nr_shuffle[j]] = [0, 0]

        # Saving test samples
        for nr_of_file in test_files:
            file = files[nr_of_file]
            name = str(file[1])
            temp_path_test = set_path_test + "\\" + str(dirs[file[0]])
            if not os.path.exists(temp_path_test):
                os.makedirs(temp_path_test)
            copy(directory + '\\' + str(dirs[file[0]]) + '\\' + name, temp_path_test)

        # Saving training samples
        for file in temp_files:
            if file != [0, 0]:
                name = str(file[1])
                temp_path = set_path + "\\" + str(dirs[file[0]])
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                copy(directory + '\\' + str(dirs[file[0]]) + '\\' + name, temp_path)

        print('Sets for iteration ', i+1, 'prepared. Saved into: ', set_path, set_path_test)
        ch_point = k_tem + ch_point
        directories.append(set_path)
        test_directories.append(set_path_test)

    return directories, test_directories


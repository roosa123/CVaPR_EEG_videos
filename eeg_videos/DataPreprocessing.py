from __future__ import division
import numpy as np
import os
import _pickle as cPickle
import matplotlib.pyplot as plt
# from scipy.stats import frechet_r_gen

import EEG.EEG.eeg as eeg
from mne.time_frequency import psd_array_welch
from scipy.integrate import simps
from scipy.interpolate import griddata


def normalize(sample):
    return 2 * (np.divide(np.subtract(sample, np.min(sample)), (np.max(sample) - np.min(sample)))) - 1


def load_data():
    path = "../DEAP/data_preprocessed_python"
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
    print(len(list_of_data))

    return list_of_labels, list_of_data, files


def preprocess_data(list_of_labels, list_of_data, files, directory):
    #  Next step is to extract features from all packages of Data

    # ============== some useful variables ==============
    channel_positions = np.array([[0, 3], [1, 3], [2, 2], [2, 0], [3, 1], [3, 3], [4, 2], [4, 0],
                                  [5, 1], [5, 3], [6, 2], [6, 0], [7, 3], [8, 3], [8, 4], [6, 4],
                                  [0, 5], [1, 5], [2, 4], [2, 6], [2, 8], [3, 7], [3, 5], [4, 4],
                                  [4, 6], [4, 8], [5, 7], [5, 5], [6, 6], [6, 8], [7, 5], [8, 5]])

    points = np.zeros((81, 2))
    x1, y1 = 0, 0

    for x in range(len(points)):
        points[x] = [x1, y1]
        y1 += 1
        if y1 > 8:
            y1 = 0
            x1 += 1

    # ===================================================

    for i in range(len(list_of_data)):
        for j in range(len(list_of_data[i])):
            avg_band_power = np.zeros((9, 9, 4))
            for k in range(len(list_of_data[i][j])):
                data = list_of_data[i][j][k]

                # kod Marcela
                # ps = 10*np.log10(np.abs(np.fft.fft(data))**2)
                # time_step = 1 / 128
                # freqs = np.fft.fftfreq(data.size, time_step)
                # N = len(freqs)
                # freqs = np.abs(freqs[0:N//2])
                # idx = np.argsort(freqs)

                # kod znaleziony
                fs = 128  # sampling rate
                freqs, ps = eeg.computeFFT(data, fs=fs)
                idx = np.argsort(freqs)

                plt.plot(freqs[idx], ps[idx])
                plt.title(label=(files[i] + " video: " + str(j + 1) + " chanel: " + str(k + 1)))
                plt.show()

                # PSD

                psds, freqs = psd_array_welch(data, sfreq=fs, n_per_seg=7, n_fft=np.shape(data)[0])

                plt.plot(freqs[1:np.shape(freqs)[0] - 1], psds[1:np.shape(psds)[0] - 1])
                plt.show()

                # 1. find average band powers (for alpha, beta, gamma and theta bands)
                freq_bands = {
                    'alpha': [8, 13],
                    'beta': [13, 30],
                    'gamma': [30, 40],
                    'theta': [4, 8]
                }

                freq_res = freqs[1] - freqs[0]
                avg_power = []

                for band in ['alpha', 'beta', 'gamma', 'theta']:
                    idx = np.logical_and(freqs >= freq_bands[band][0], freqs <= freq_bands[band][1])
                    avg = simps(psds[idx], dx=freq_res)
                    avg_power.append(avg)

                avg_band_power[channel_positions[k][0]][channel_positions[k][1]] = avg_power

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

            avg_bands = avg_band_power.reshape((81, 4))
            nx = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), num=120)
            ny = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=120)
            grid_x, grid_y = np.meshgrid(nx, ny)

            out = []
            for band in range(4):
                grid = griddata(points, avg_bands[:, band], (grid_x, grid_y), fill_value=0.0, method='cubic')
                out.append(grid)

            cur_dir = directory + str(list_of_labels[i][j][1])

            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            a = np.array(out)
            img = np.rollaxis(a, 2)
            img = np.rollaxis(img, 2)

            np.save(cur_dir + '\\' + str(i) + str(j) + '.npy', np.array(img))

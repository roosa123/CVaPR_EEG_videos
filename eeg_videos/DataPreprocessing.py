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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


path = "../DEAP/data_preprocessed_python"
List_of_labels = []
List_of_data = []
files = []

for (_, _, sets) in os.walk(path):
    files.extend(sets)

for file in files:

    Participant_base = cPickle.load(open(path + '/' + file, 'rb'), encoding='latin1')
    # ----------Labels - four discrete  states---------------

    Labels = Participant_base['labels']
    N, _ = Labels.shape
    Labels2 = np.zeros([N, 2])

    # States - labels IMPORTANT
    # 0 - Meditation  Val > 5 Arousal < 5
    # 1 - Boredom     Val < 5 Arousal < 5
    # 2 - Excitement  Val > 5 Arousal > 5
    # 3 - Frustration Val < 5 Arousal > 5

    for i in range(1, N + 1):
        Labels2[i - 1, 0] = i
        curr_valence = Labels[i - 1, 0]
        curr_arousal = Labels[i - 1, 1]
        if curr_valence > 5:
            if curr_arousal <= 5:
                Labels2[i - 1, 1] = 0
            else:
                Labels2[i - 1, 1] = 2
        else:
            if curr_arousal <= 5:
                Labels2[i - 1, 1] = 1
            else:
                Labels2[i - 1, 1] = 3

    # -------------- Lists for labels and data------------------

    Data = Participant_base['data']
    List_of_labels.append(Labels2)
    List_of_data.append(np.delete(Data, np.s_[32:], axis=1))
    # ----------------------------------------------------------

print(len(List_of_labels))
print(len(List_of_data))

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

directory = '..\\DEAP\\img\\'

# ===================================================

train_data = ImageDataGenerator(
    rotation_range=120,
    horizontal_flip=True,
    height_shift_range=0.6,
    width_shift_range=0.6,
).flow_from_directory(
    '..\\DEAP\\img',
    target_size=(9, 9),
    batch_size=8
)

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 4)))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit_generator(
#     train_data,
#     steps_per_epoch=64,
#     epochs=3,
#     validation_steps=2
# )

for i in range(len(List_of_data)):
    for j in range(len(List_of_data[i])):
        avg_band_power = np.zeros((9, 9, 4))
        for k in range(len(List_of_data[i][j])):
            data = List_of_data[i][j][k]

            # kod Marcela
            # ps = 10*np.log10(np.abs(np.fft.fft(data))**2)
            # time_step = 1 / 128
            # freqs = np.fft.fftfreq(data.size, time_step)
            # N = len(freqs)
            # freqs = np.abs(freqs[0:N//2])
            # idx = np.argsort(freqs)

            # kod znaleziony
            fs = 128        # sampling rate
            freqs, ps = eeg.computeFFT(data, fs=fs)
            idx = np.argsort(freqs)

            # plt.plot(freqs[idx], ps[idx])
            # plt.title(label=(files[i] + " video: " + str(j + 1) + " chanel: " + str(k + 1)))
            # plt.show()

            # PSD

            psds, freqs = psd_array_welch(data, sfreq=fs, n_per_seg=7, n_fft=np.shape(data)[0])

            # plt.plot(freqs[1:np.shape(freqs)[0] - 1], psds[1:np.shape(psds)[0] - 1])
            # plt.show()

            # 1. find average band powers (for alpha, beta, gamma and theta bands)
            freq_bands = {
                'alpha':    [8, 13],
                'beta':     [13, 30],
                'gamma':    [30, 40],
                'theta':    [4, 8]
            }

            freq_res = freqs[1] - freqs[0]
            avg_power = []

            for band in ['alpha', 'beta', 'gamma', 'theta']:
                idx = np.logical_and(freqs >= freq_bands[band][0], freqs <= freq_bands[band][1])
                avg = simps(psds[idx], dx=freq_res)
                avg_power.append(avg)

            avg_band_power[channel_positions[k][0]][channel_positions[k][1]] = avg_power

            # def func(x, y):
            #     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
            #
            #
            # grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
            # points = np.random.rand(1000, 2)
            # values = func(points[:, 0], points[:, 1])
            #
            # from scipy.interpolate import griddata
            # grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
            # grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
            # grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
            #
            # print(np.max(grid_z0))
            # print(np.max(grid_z1))
            # print(np.max(grid_z2))
            #
            # print(np.min(grid_z0))
            # print(np.min(grid_z1))
            # print(np.min(grid_z2))
            #
            # plt.subplot(221)
            # plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')
            # plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
            # plt.title('Original')
            # plt.subplot(222)
            # plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Nearest')
            # plt.subplot(223)
            # plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Linear')
            # plt.subplot(224)
            # plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
            # plt.title('Cubic')
            # plt.gcf().set_size_inches(6, 6)
            # plt.show()
            #
            # pass

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

        cur_dir = directory + str(List_of_labels[i][j][1])

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        np.save(cur_dir + '\\' + str(i) + str(j) + '.npy', np.array(out))

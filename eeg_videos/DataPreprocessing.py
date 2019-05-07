from __future__ import division
import numpy as np
import os
import _pickle as cPickle
import matplotlib.pyplot as plt
import EEG.EEG.eeg as eeg
from mne.time_frequency import psd_array_welch


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
    List_of_data.append(Data)
    # ----------------------------------------------------------

print(len(List_of_labels))
print(len(List_of_data))

#  Next step is to extract features from all packages of Data

for i in range(len(List_of_data) - 1):
    for j in range(len(List_of_data[i]) - 1):
        for k in range(len(List_of_data[i][j]) - 1):
            data = List_of_data[i][j][k]

            # kod Marcela
            # ps = 10*np.log10(np.abs(np.fft.fft(data))**2)
            # time_step = 1 / 128
            # freqs = np.fft.fftfreq(data.size, time_step)
            # N = len(freqs)
            # freqs = np.abs(freqs[0:N//2])
            # idx = np.argsort(freqs)

            # kod znaleziony
            freqs, ps = eeg.computeFFT(data, fs=128)
            idx = np.argsort(freqs)

            plt.plot(freqs[idx], ps[idx])
            plt.title(label=(files[i] + " video: " + str(j + 1) + " chanel: " + str(k + 1)))
            plt.show()

            psd_array_welch(data, )
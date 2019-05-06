import numpy as np
import os

path = "../data_preprocessed_python"

# Lists of labels and data for all participants
List_of_labels = []
List_of_data = []
files = []

for (_, _, sets) in os.walk(path):
    files.extend(sets)

for file in files:

    Participant_base = np.load(path + '/' + file, encoding='bytes', allow_pickle=True)

    # ----------Labels - four discrete  states---------------

    Labels = Participant_base[b'labels']
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

    Data = Participant_base[b'data']

    List_of_labels.append(Labels2)
    List_of_data.append(Data)
    # ----------------------------------------------------------

print(len(List_of_labels))
print(len(List_of_data))

#  Next step is to extract features from all packages of Data
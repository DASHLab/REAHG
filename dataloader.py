import numpy as np
from loss_and_metric import *


class DataLoader():

    def __init__(self, data, batch_size=4, seq_length=16):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.load_preprocessed(data)

        self.reset_batch_pointer()


    def load_preprocessed(self, data):

        pedsList_data = []
        numPeds_data = []
        for i in range(len(data)):
            person_id = []
            for j in range(data.shape[1]):
                if (data[i, j, 0] != 0 and data[i, j, 1] != 0):
                    person_id.append(j)
                else:
                    person_id.append(-1)

            pedsList_data.append(person_id)
            numPeds_data.append(len(person_id))

        self.data = data
        self.numPedsList = numPeds_data
        self.pedsList = pedsList_data
        all_frame_data = self.data

        num_seq_in_dataset = int(len(all_frame_data) / (self.seq_length))
        self.num_batches = int(num_seq_in_dataset / self.batch_size)


    def next_batch(self):


        x_batch = []

        PedsList_batch = []

        # Iteration index
        i = 0
        while i < self.batch_size:

            frame_data = self.data
            pedsList = self.pedsList

            idx = self.frame_pointer

            if idx + self.seq_length - 1 < len(frame_data):

                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]

                x_batch.append(seq_source_frame_data)
                PedsList_batch.append(seq_PedsList)

                self.frame_pointer += self.seq_length

                i += 1

        return x_batch, PedsList_batch


    def reset_batch_pointer(self):
        self.frame_pointer = 0






def load_data(data_name):
    if data_name == "BJ":
        data = np.load("T_Drive_data.npy")
        data = data[:, 2000:3000, :]
        # Remove the parts without data
        data = data[64:-32]
        # Standardization
        index = data[:, :, 0] != 0
        g = np.zeros((data[index].shape[0], 2))
        g[:, 0] = (data[index][:, 0] - data[index][:, 0].mean()) / data[index][:, 0].std()
        g[:, 1] = (data[index][:, 1] - data[index][:, 1].mean()) / data[index][:, 1].std()
        data[index] = g

        '''
        train: 5days
        valid: 1day
        test: 1day
        '''
        train_data = data[:416]
        test_data = data[416:512]
        valid_data = data[512:]

    if data_name == "POI":
        data = np.load("Foursquare_data.npy")
        data = data[:, 1000:2000, :]
        # Standardization
        index = data[:, :, 0] != 0
        g = np.zeros((data[index].shape[0], 2))
        g[:, 0] = (data[index][:, 0] - data[index][:, 0].mean()) / data[index][:, 0].std()
        g[:, 1] = (data[index][:, 1] - data[index][:, 1].mean()) / data[index][:, 1].std()
        data[index] = g


        train_data = data[:208]
        test_data = data[208:272]
        valid_data = data[272:288]


    if data_name == "SF":
        data = np.load("T_SF_data.npy")
        # Remove the parts without data
        data = data[304:-720]
        # Standardization
        index = data[:, :, 0] != 0
        g = np.zeros((data[index].shape[0], 2))
        g[:, 0] = (data[index][:, 0] - data[index][:, 0].mean()) / data[index][:, 0].std()
        g[:, 1] = (data[index][:, 1] - data[index][:, 1].mean()) / data[index][:, 1].std()
        data[index] = g


        train_data = data[(14)*720-304:(14+3)*720-304]
        valid_data = data[11936:12368]
        test_data = data[12368:12800]

    return train_data, valid_data, test_data


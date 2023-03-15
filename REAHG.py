import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

from AHG import *
from loss_and_metric import getCoef_formodel, sample_gaussian_2d_formodel


class REAHG(nn.Module):

    def __init__(self, args, node_num):
        super(REAHG, self).__init__()

        self.args = args
        self.use_cuda = args.use_cuda
        self.rnn_size = args.rnn_size
        self.embedding_size = args.medium_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.seq_length = args.seq_length
        self.gru = args.gru
        self.num_node = node_num
        self.assign_dim = args.assign_dim

        # The LSTM cell
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        if self.gru:
            self.cell = nn.GRUCell(2 * self.embedding_size, self.rnn_size)

        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        self.tensor_embedding_layer = nn.Linear(self.rnn_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.ahgpooling = Attentional_Hierarchical_Graph(self.rnn_size, self.assign_dim)
        self.update1 = nn.Linear(2 * self.rnn_size, self.rnn_size)

    def forward(self, state, node_features, *args):
        input_data = args[0]
        hidden_states = args[1]
        cell_states = args[2]
        if self.gru:
            cell_states = None
        PedsList = args[3]

        if state == "train":
            numNodes = input_data.shape[1]
            outputs = Variable(torch.zeros((self.seq_length-1) * numNodes, self.output_size))
            if self.use_cuda:
                outputs = outputs.cuda()

            # empty: used to filter sequences with no data
            empty = 0

            for framenum, frame in enumerate(input_data):

                # Selecting users present in the current frame
                nodeIDs = [int(nodeID) for nodeID in PedsList[framenum][torch.where(PedsList[framenum] > -1)]]

                if len(nodeIDs) == 0:
                    # If no users, then go to the next frame
                    empty = empty + 1
                    continue

                corr_index = Variable((torch.LongTensor(nodeIDs)))

                if self.use_cuda:
                    corr_index = corr_index.cuda()

                nodes_current = torch.index_select(frame, 0, corr_index)
                if self.use_cuda:
                    nodes_current = nodes_current.cuda()

                hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

                #Update node features in graph
                x = self.update1(torch.cat([node_features.clone().detach(),hidden_states.clone().detach()],dim=1))
                x1 = hidden_states.clone().detach()

                #Adaptive graph generation and conducting attentional hierarchical graph learning
                adj = F.softmax(F.relu(torch.mm(x, x.transpose(0, 1))), dim=1)
                r = self.ahgpooling(adj, x1)

                if not self.gru:
                    cell_states_current = torch.index_select(cell_states, 0, corr_index)

                input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
                r_1 = torch.index_select(r, 0, corr_index)
                tensor_embedded = self.relu(self.tensor_embedding_layer(r_1))

                concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                if not self.gru:
                    h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                else:
                    h_nodes = self.cell(concat_embedded, (hidden_states_current))

                outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

                # Update hidden and cell states
                hidden_states[corr_index.data] = h_nodes
                if not self.gru:
                    cell_states[corr_index.data] = c_nodes

            # Reshape outputs
            outputs_return = Variable(torch.zeros(self.seq_length-1, numNodes, self.output_size))
            if self.use_cuda:
                outputs_return = outputs_return.cuda()

            for framenum in range(self.seq_length-1):
                for node in range(numNodes):
                    outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]

            # if sequences with no data
            if empty == 15:
                adj = F.softmax(F.relu(torch.mm(node_features.clone().detach(), node_features.clone().detach().transpose(0, 1))), dim=1)
                return outputs_return, hidden_states, cell_states, node_features, adj
            else:
                return outputs_return, hidden_states, cell_states, x, adj



        # input: sequence[0:n/2]
        # predict: sequence[n/2:]
        # "valid" is different from "train"
        if state == "valid":

            numNodes = input_data.shape[1]
            outputs = Variable(torch.zeros((self.seq_length -1) * numNodes, self.output_size))
            if self.use_cuda:
                outputs = outputs.cuda()

            for framenum, frame in enumerate(input_data):

                # sequence[0:n/2]
                if framenum < (self.seq_length/2):

                    #uers present in the current frame
                    nodeIDs = [int(nodeID) for nodeID in PedsList[framenum][torch.where(PedsList[framenum] > -1)]]

                    if len(nodeIDs) == 0:
                        # If no users, then go to the next frame
                        continue

                    corr_index = Variable((torch.LongTensor(nodeIDs)))

                    if self.use_cuda:
                        corr_index = corr_index.cuda()

                    nodes_current = torch.index_select(frame, 0, corr_index)
                    if self.use_cuda:
                        nodes_current = nodes_current.cuda()

                    hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

                    # Update node features in graph
                    x = self.update1(torch.cat([node_features.clone().detach(), hidden_states.clone().detach()], dim=1))
                    x1 = hidden_states.clone().detach()

                    adj = F.softmax(F.relu(torch.mm(x, x.transpose(0, 1))), dim=1)
                    r = self.ahgpooling(adj, x1)

                    if not self.gru:
                        cell_states_current = torch.index_select(cell_states, 0, corr_index)

                    input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
                    r_1 = torch.index_select(r, 0, corr_index)
                    tensor_embedded = self.relu(self.tensor_embedding_layer(r_1))

                    concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                    if not self.gru:
                        h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                    else:
                        h_nodes = self.cell(concat_embedded, (hidden_states_current))

                    outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

                    # Update hidden and cell states
                    hidden_states[corr_index.data] = h_nodes
                    if not self.gru:
                        cell_states[corr_index.data] = c_nodes

                # sequence[n/2:]
                else:

                    nodeIDs = [int(nodeID) for nodeID in PedsList[int(self.seq_length/2 - 1)][torch.where(PedsList[int(self.seq_length/2 - 1)] > -1)]]

                    if len(nodeIDs) == 0:
                        # If no users, then break
                        break

                    corr_index = Variable((torch.LongTensor(nodeIDs)))

                    if self.use_cuda:
                        corr_index = corr_index.cuda()

                    # Calculate the predicted coordinates based on the output of the previous frame
                    last_pred = Variable(torch.zeros(numNodes, self.output_size))
                    for node in range(numNodes):
                        last_pred[node, :] = outputs[(framenum - 1) * numNodes + node, :]
                    if self.use_cuda:
                        last_pred = last_pred.cuda()
                    mux, muy, sx, sy, corr = getCoef_formodel(last_pred)
                    next_x, next_y = sample_gaussian_2d_formodel(mux.data, muy.data, sx.data, sy.data, corr.data)
                    next_vals = torch.DoubleTensor(numNodes, 2)
                    next_vals[:, 0] = next_x
                    next_vals[:, 1] = next_y
                    if self.use_cuda:
                        next_vals = next_vals.cuda()


                    nodes_current = torch.index_select(next_vals, 0, corr_index)
                    if self.use_cuda:
                        nodes_current = nodes_current.cuda()

                    hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

                    # Update node features in graph
                    x = node_features.clone().detach()
                    x1 = hidden_states.clone().detach()


                    adj = F.softmax(F.relu(torch.mm(x, x.transpose(0, 1))), dim=1)
                    r = self.ahgpooling(adj, x1)


                    if not self.gru:
                        cell_states_current = torch.index_select(cell_states, 0, corr_index)

                    input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
                    r_1 = torch.index_select(r, 0, corr_index)
                    tensor_embedded = self.relu(self.tensor_embedding_layer(r_1))
                    concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

                    if not self.gru:
                        h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
                    else:
                        h_nodes = self.cell(concat_embedded, (hidden_states_current))

                    outputs[framenum * numNodes + corr_index.data] = self.output_layer(h_nodes)

                    # Update hidden and cell states
                    hidden_states[corr_index.data] = h_nodes
                    if not self.gru:
                        cell_states[corr_index.data] = c_nodes

            # Reshape outputs
            outputs_return = Variable(torch.zeros(self.seq_length - 1, numNodes, self.output_size))
            if self.use_cuda:
                outputs_return = outputs_return.cuda()
            for framenum in range(self.seq_length - 1):
                for node in range(numNodes):
                    outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]

            return outputs_return, hidden_states, cell_states, x, adj

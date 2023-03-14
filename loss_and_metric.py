import numpy as np
import torch
from torch.autograd import Variable

import os
import shutil
from os import walk
import math




def getCoef(outputs):

    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def getCoef_formodel(outputs):

    mux, muy, sx, sy, corr = outputs[ :, 0], outputs[ :, 1], outputs[ :, 2], outputs[ :, 3], outputs[ :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent):

    o_mux, o_muy, o_sx, o_sy, o_corr = mux[:, :], muy[:, :], sx[:, :], sy[:, :], corr[:, :]

    seq_legth = mux.size()[0]
    numNodes = mux.size()[1]

    next_x = torch.zeros([seq_legth,numNodes])
    next_y = torch.zeros([seq_legth,numNodes])

    for t in range(seq_legth):
        for node in range(numNodes):
            mean = [o_mux[t,node].detach().cpu(), o_muy[t,node].detach().cpu()]
            cov = [[o_sx[t,node].detach().cpu() * o_sx[t,node].detach().cpu(), o_corr[t,node].detach().cpu() * o_sx[t,node].detach().cpu() * o_sy[t,node].detach().cpu()],
                   [o_corr[t,node].detach().cpu() * o_sx[t,node].detach().cpu() * o_sy[t,node].detach().cpu(), o_sy[t,node].detach().cpu() * o_sy[t,node].detach().cpu()]]

            mean = np.array(mean, dtype='float')
            cov = np.array(cov, dtype='float')

            next_values = np.random.multivariate_normal(mean, cov, 1)
            next_x[t,node] = next_values[0][0]
            next_y[t,node] = next_values[0][1]

    return next_x, next_y


def sample_gaussian_2d_formodel(mux, muy, sx, sy, corr):

    o_mux, o_muy, o_sx, o_sy, o_corr = mux[:], muy[:], sx[:], sy[:], corr[:]

    numNodes = mux.size()[0]
    next_x = torch.zeros([numNodes])
    next_y = torch.zeros([numNodes])

    for node in range(numNodes):
        mean = [o_mux[node].detach().cpu(), o_muy[node].detach().cpu()]
        cov = [[o_sx[node].detach().cpu() * o_sx[node].detach().cpu(), o_corr[node].detach().cpu() * o_sx[node].detach().cpu() * o_sy[node].detach().cpu()],
                   [o_corr[node].detach().cpu() * o_sx[node].detach().cpu() * o_sy[node].detach().cpu(), o_sy[node].detach().cpu() * o_sy[node].detach().cpu()]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda):

    pred_length = ret_nodes.size()[0]

    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for tstep in range(pred_length):
        counter = 0

        for nodeID in assumedNodesPresent[torch.where(assumedNodesPresent > -1)]:

            nodeID = int(nodeID)
            if nodeID not in trueNodesPresent[tstep]:
                continue

            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error), error




def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent):
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[torch.where(assumedNodesPresent > -1)]:
        nodeID = int(nodeID)

        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]

        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1

    if counter != 0:
        error = error / counter

    return error

def get_t_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent,t):
    error = 0
    counter = 0

    # Last time-step
    tstep = t
    for nodeID in assumedNodesPresent[torch.where(assumedNodesPresent > -1)]:
        nodeID = int(nodeID)

        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]

        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1

    if counter != 0:
        error = error / counter

    return error





def Gaussian2DLikelihood(outputs, targets, nodesPresent):

    seq_length = outputs.size()[0]
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(seq_length):

        nodeIDs = [int(nodeID) for nodeID in nodesPresent[framenum][torch.where(nodesPresent[framenum] > -1)]]
        nodeIDs1 = [int(nodeID) for nodeID in nodesPresent[framenum+1][torch.where(nodesPresent[framenum+1] > -1)]]

        for nodeID in nodeIDs:
            if(nodeID in nodeIDs1):
                loss = loss + result[framenum, nodeID]
                counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss



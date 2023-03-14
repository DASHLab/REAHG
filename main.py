import torch
torch.cuda.current_device()
import argparse
import time
from tqdm import tqdm

from A2HiPool_model import A2HiPool
from dataloader import DataLoader, load_data
from loss_and_metric import *
torch.set_default_tensor_type(torch.DoubleTensor)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=2)

    parser.add_argument('--output_size', type=int, default=5)

    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')

    parser.add_argument('--seq_length', type=int, default=16,
                        help='RNN sequence length')

    parser.add_argument('--num_epochs', type=int, default=60,
                        help='number of epochs')

    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability')

    parser.add_argument('--medium_size', type=int, default=64,
                        help='Medium dimension for the spatial coordinates')

    parser.add_argument('--assign_dim', type=int, default=500,
                        help='Assign_dim in pooling')

    parser.add_argument('--lambda_param', type=float, default=0.0000,
                        help='L2 regularization parameter')

    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')

    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')

    parser.add_argument('--data_name', type=str, default="POI",
                        help='POI : Foursquare datasets'
                             'BJ: T-Drive datasets'
                             'SF: T-SF datasets')

    args = parser.parse_args()

    train(args)







def train(args):

    train_data, valid_data, test_data = load_data(args.data_name)
    train_dataloader = DataLoader(train_data, args.batch_size, args.seq_length)
    valid_dataloader = DataLoader(valid_data, args.batch_size, args.seq_length)
    test_dataloader = DataLoader(test_data, args.batch_size, args.seq_length)

    # model creation
    node_num = train_data.shape[1]
    net = A2HiPool(args, node_num)
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), weight_decay = args.lambda_param)

    best_val_loss = 100

    smallest_err_val = 100000

    best_epoch_val = 0

    best_err_epoch_val = 0

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')

        train_dataloader.reset_batch_pointer()

        loss_epoch = 0

        node_features = Variable(torch.zeros(node_num, args.rnn_size), requires_grad=True)

        if args.use_cuda:
            node_features = node_features.cuda()

        for batch in tqdm(range(train_dataloader.num_batches)):
            start = time.time()

            x, PedsList = train_dataloader.next_batch()

            loss_batch = 0

            for sequence in range(train_dataloader.batch_size):

                x_seq, PedsList_seq = x[sequence], PedsList[sequence]

                x_seq = Variable(torch.from_numpy(x_seq))

                PedsList_seq = torch.from_numpy(np.array(PedsList_seq))

                if args.use_cuda:
                    x_seq = x_seq.cuda()
                    PedsList_seq = PedsList_seq.cuda()

                numNodes = x_seq.shape[1]

                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()

                net.zero_grad()

                optimizer.zero_grad()

                outputs, hidden_states1, _, node_features, adj1 = net("train", node_features, x_seq[:-1], hidden_states, cell_states, PedsList_seq[:-1])

                loss = Gaussian2DLikelihood(outputs, x_seq[1:], PedsList_seq)

                if loss == 0:
                    continue

                loss_batch = loss.item() + loss_batch

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                optimizer.step()

            end = time.time()
            loss_batch = loss_batch / train_dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch /= train_dataloader.num_batches

        print("Training epoch: " + str(epoch) + " loss: " + str(loss_epoch) + '\n')




        print('****************Validation epoch beginning******************')

        valid_dataloader.reset_batch_pointer()

        loss_epoch = 0

        errors = []
        FDE = []
        DE2 = []
        DE6 = []

        for batch in tqdm(range(valid_dataloader.num_batches)):

            x, PedsList = valid_dataloader.next_batch()

            loss_batch = 0

            for sequence in range(valid_dataloader.batch_size):
                x_seq, PedsList_seq = x[sequence], PedsList[sequence]

                x_seq = Variable(torch.from_numpy(x_seq))
                PedsList_seq = torch.from_numpy(np.array(PedsList_seq))

                if args.use_cuda:
                    x_seq = x_seq.cuda()
                    PedsList_seq = PedsList_seq.cuda()

                numNodes = x_seq.shape[1]

                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()

                outputs, _, _,_,_ = net("valid", node_features, x_seq[:-1], hidden_states, cell_states, PedsList_seq[:-1])

                loss = Gaussian2DLikelihood(outputs, x_seq[1:], PedsList_seq)

                mux, muy, sx, sy, corr = getCoef(outputs)

                next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, PedsList_seq[1:],)
                next_vals = torch.FloatTensor(outputs.shape[0], numNodes, 2)
                next_vals[:, :, 0] = next_x
                next_vals[:, :, 1] = next_y
                if args.use_cuda:
                    next_vals = next_vals.cuda()

                seq_length = x_seq.shape[0]
                medium = int(seq_length / 2 - 1)

                err, error_all = get_mean_error(next_vals[medium:], x_seq[medium + 1:], PedsList_seq[medium], PedsList_seq[medium + 1:], args.use_cuda)

                errors.append(err.data.cpu().numpy().tolist())
                FDE.append(error_all[-1].data.cpu().numpy().tolist())
                DE2.append(error_all[1].data.cpu().numpy().tolist())
                DE6.append(error_all[5].data.cpu().numpy().tolist())

                loss_batch += loss.item()

            loss_batch = loss_batch / valid_dataloader.batch_size
            loss_epoch += loss_batch

        if valid_dataloader.num_batches != 0:
            loss_epoch = loss_epoch / valid_dataloader.num_batches

            ckpt_dict = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

            # Update best validation loss and ADE
            if loss_epoch < best_val_loss:
                best_val_loss = loss_epoch
                best_epoch_val = epoch

            ADE = np.mean(errors)
            if ADE < smallest_err_val:
                smallest_err_val = ADE
                best_err_epoch_val = epoch
                torch.save(ckpt_dict, 'best_checkpoint.pth.tar')

            print('(epoch {}), valid_loss = {:.3f}, valid_err = {:.3f}'.format(epoch, loss_epoch, ADE))
            print('Best epoch', best_epoch_val, 'Best validation loss', best_val_loss, 'Best error epoch', best_err_epoch_val, 'Best error', smallest_err_val)
            fde = np.array(FDE)[np.array(FDE)>0]
            print("ade:",np.mean(errors))
            print("Fde:", np.mean(fde))
            print("de@2", np.mean(np.array(DE2)[np.array(DE2)>0]), "de@6", np.mean(np.array(DE6)[np.array(DE6)>0]))


    print('****************test epoch beginning******************')

    test_dataloader.reset_batch_pointer()

    t_loss_epoch = 0

    t_errors = []
    t_FDE = []
    t_DE2 = []
    t_DE6 = []

    for batch in tqdm(range(test_dataloader.num_batches)):

        x, PedsList = test_dataloader.next_batch()

        t_loss_batch = 0

        for sequence in range(test_dataloader.batch_size):

            x_seq, PedsList_seq = x[sequence], PedsList[sequence]

            x_seq = Variable(torch.from_numpy(x_seq))

            PedsList_seq = torch.from_numpy(np.array(PedsList_seq))

            if args.use_cuda:
                x_seq = x_seq.cuda()
                PedsList_seq = PedsList_seq.cuda()

            numNodes = x_seq.shape[1]

            hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
            if args.use_cuda:
                hidden_states = hidden_states.cuda()

            cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()

            ckpt = torch.load('best_checkpoint.pth.tar')
            net.load_state_dict(ckpt['state_dict'])

            outputs, _, _, _, _ = net("valid", node_features, x_seq[:-1], hidden_states, cell_states, PedsList_seq[:-1])

            loss = Gaussian2DLikelihood(outputs, x_seq[1:], PedsList_seq)

            mux, muy, sx, sy, corr = getCoef(outputs)

            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, PedsList_seq[1:], )
            next_vals = torch.FloatTensor(outputs.shape[0], numNodes, 2)
            next_vals[:, :, 0] = next_x
            next_vals[:, :, 1] = next_y
            if args.use_cuda:
                next_vals = next_vals.cuda()

            seq_length = x_seq.shape[0]

            medium = int(seq_length / 2 - 1)

            err, error_all = get_mean_error(next_vals[medium:], x_seq[medium + 1:], PedsList_seq[medium],
                                            PedsList_seq[medium + 1:],
                                            args.use_cuda)

            t_errors.append(err.data.cpu().numpy().tolist())
            t_FDE.append(error_all[-1].data.cpu().numpy().tolist())
            t_DE2.append(error_all[1].data.cpu().numpy().tolist())
            t_DE6.append(error_all[5].data.cpu().numpy().tolist())

            t_loss_batch += loss.item()

        t_loss_batch = t_loss_batch / test_dataloader.batch_size
        t_loss_epoch += t_loss_batch

    if test_dataloader.num_batches != 0:
        t_loss_epoch = t_loss_epoch / test_dataloader.num_batches

        t_fde = np.array(t_FDE)[np.array(t_FDE) > 0]
        print("ade:", np.mean(t_errors))
        print("Fde:", np.mean(t_fde))
        print("de@2", np.mean(np.array(t_DE2)[np.array(t_DE2) > 0]), "de@6",
              np.mean(np.array(t_DE6)[np.array(t_DE6) > 0]))







if __name__ == '__main__':
    main()
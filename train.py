import os
import torch
from torch import nn
import argparse
from src.DataLoader import Dataset_Hilbert, contruct_dataloader_from_disk
from src.AutoEncoder import autoencoder


def get_args():

    parser = argparse.ArgumentParser('Train Contact-Map AutoEncoder')
    parser.add_argument('--hdf5_file', type=str, help='Path to HDF5 file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to Checkpoint Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--early_stop', type=int, default=40, help='Early stop limit')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay to optimizer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--nc', type=int, default=1, help='Number of channels in data')
    parser.add_argument('--ld', type=int, default=256, help='latent dimension size')

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    return args


def create_folders():

    if not os.path.exists("./output/"):

        os.mkdir("./output/")


def train(args):

    nc = args.nc

    ndf = args.ld

    model = autoencoder(nc, ndf).to('cuda:1')

    checkpoint = args.checkpoint

    if checkpoint is not None and os.path.exists(checkpoint):

        model.load_state_dict(torch.load(checkpoint))

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = contruct_dataloader_from_disk(args.hdf5_file, args.batch_size)

    num_epochs = args.epochs

    early_stop_limit = args.early_stop

    early_stop_count = 0

    train_loss = []

    create_folders()

    best_path = "./output/HILBERT_AE_best.pth"

    for epoch in range(num_epochs):

        loss_train = 0

        for idx, minibatch_ in enumerate(train_loader):

            seq, hilbert_map = minibatch_

            hilbert_map = torch.stack(hilbert_map).permute(0, 3, 1, 2).type(torch.FloatTensor)

            hilbert_map = hilbert_map.to('cuda:1')

            # ===================forward=====================

            output = model(hilbert_map)

            loss = criterion(output, hilbert_map)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss_train.item() / idx))

        train_loss.append(loss_train.item() / idx)

        if epoch % 10 == 0:

            torch.save(model.state_dict(), "./output/HILBERT_AE_{}.pth".format(epoch))

        if len(train_loss) > 2 and train_loss[-1] == min(train_loss):

            torch.save(model.state_dict(), best_path)

            early_stop_count = 0

        else:

            early_stop_count += 1

        if early_stop_count > early_stop_limit:

            break

    print("AutoEncoder was trained !!")


if __name__ == '__main__':

    args = get_args()

    train(args)

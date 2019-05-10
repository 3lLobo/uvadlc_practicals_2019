import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        input_dim = 28*28

        net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU,
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_dim, 2*z_dim))
        # Need to init?
        torch.nn.init.xavier_uniform(net.weight)
        net.bias.data.fill_(0.01)
        self.net = net

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.

        Of q distribution.
        x is input image space.
        """

        flat = input.view(-1, 28*28)
        flat = flat.squeeze()
        out = self.net(flat)
        mean = out[:self.z_dim]
        std = out[self.z_dim:]
        ident = torch.diag(torch.ones_like(self.z_dim))
        std = ident @ std

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        output_dim = 28 * 28
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        input_dim = z_dim

        net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                            nn.ReLU,
                            nn.Dropout(0.2),
                            nn.Linear(hidden_dim, output_dim),
                            nn.Sigmoid)
        # Need to init?
        torch.nn.init.xavier_uniform(net.weight)
        net.bias.data.fill_(0.01)
        self.net = net

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        mean = self.net(input)
        mean = mean.view(-1, 784)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        enco_mean, enco_std = self.encoder(input)
        latent_mean = torch.zeros(self.z_dim).view(1, self.z_dim)
        latent_std = torch.diag(torch.ones(self.z_dim)).view(1, self.z_dim, self.z_dim)

        KL = (torch.log(torch.sqrt(enco_std) / torch.sqrt(latent_std)) +
              (latent_std + (latent_mean - enco_mean).pow(2) / 2 * enco_std) - 1/2)

        # Calculate latent space
        eps = torch.randn(self.z_dim)

        latent_space = eps * latent_std + latent_mean

        deco_mean = self.decoder(latent_space)
        self.deco_mean = deco_mean

        average_negative_elbo = - 1/784 * torch.sum(
            nn.functional.binary_cross_entropy(deco_mean, input) - KL)


        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # TODO: Bernuli distribution from deco_mean sample 20 times
        im_means = self.deco_mean
        sampled_ims, im_means = None, None

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = None
    raise NotImplementedError()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()

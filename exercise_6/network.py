import torch.nn as nn


def to_img(x):
    """ Maps a 2D tensor from range [-1, 1] to 4D tensor with range [0, 1].
    Useful for plotting of reconstructions.

    :param x: 2D Tensor that is supposed to be converted
    :return: Converted 4D Tensor with b, c, w, h, where w = h = 28
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def add_white_noise(x, factor=0.5, stddev=1):
    """ Adds white noise to an input tensor.
    To make sure that data is in intended range [min, max], use torch.clamp(x, min, max) after applying this function.

    :param x: ND Tensor that is altered
    :param factor: A factor that controls the strength of the additive noise
    :param stddev: The stddev of the normal distribution used for generating the noise
    :return: ND Tensor, x with white noise
    """
    # add white noise to tensor
    noise = x.clone().normal_(0, stddev)
    return x + noise * factor


class Autoencoder(nn.Module):

    def __init__(self, input_shape=(28, 28)):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ########## YOUR CODE HERE #############
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            ########## YOUR CODE HERE #############
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        ########## YOUR CODE HERE #############
        h = self.encoder(x)
        x = self.decoder(h)
        return x


class Conv_Autoencoder(nn.Module):
    def __init__(self):
        super(Conv_Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
           ########## YOUR CODE HERE #############
           nn.Conv2d(in_channels=1,out_channels=4,kernel_size=5), # 28x28x1 to 24x24x4
           nn.ReLU(), # stays 24x24x6
           nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5), # 24x24x6 to 20x20x8 
           nn.ReLU(),
           nn.Flatten(), # 20x20x8 to 3200
           nn.Linear(3200,10),
           nn.Softmax()
            )
        
        self.decoder = nn.Sequential(
           ########## YOUR CODE HERE #############
           nn.Linear(10,400),
           nn.ReLU(),
           nn.Linear(400,4000),
           nn.Unflatten(1,(8,25,20)),
           nn.ConvTranspose2d(in_channels=10,out_channels=10,kernel_size=5),
           nn.ReLU(),
           nn.ConvTranspose2d(in_channels=10,out_channels=1,kernel_size=5),
           nn.Tanh()
            )
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        ########## YOUR CODE HERE #############
        x = x.view(-1, 28*28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
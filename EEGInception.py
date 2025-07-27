import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from alignment_techniques import LatentAlignment2d, AdaptiveBatchNorm2d, EuclideanAlignment


class EEGInception(nn.Module):
    """
    EEGInception implementation based on
    Santamaria-Vazquez, E., Martinez-Cagigal, V., Vaquerizo-Villar, F. and Hornero, R., 2020.
    EEG-inception: a novel deep convolutional neural network for assistive ERP-based brain-computer interfaces.
    IEEE Transactions on Neural Systems and Rehabilitation Engineering, 28(12), pp.2773-2782.
    https://ieeexplore.ieee.org/abstract/document/9311146
    Assumes sampling frequency of 100 for kernel size choices. Uses odd kernel sizes for valid paddings.
    """
    '''
    classe che definisce un'architettura di rete neurale convoluzionale (CNN) ispirata ai moduli
    Inception, comunemente usati in visione artificiale, ma adattati per i segnali EEG. L'obiettivo
    è classificare i segnali EEG, probabilmente per compiti come lo speller P300, dove si cerca una
    risposta cerebrale specifica a stimoli bersaglio
    '''
    #costruttore del modello
    def __init__(self, in_shape, n_out, alignment='None', dropout=0.25):
        '''
            Args:
            in-shape: tupla o lista che specifica le dimensioni del segnale EEG in ingresso, tipicamente (numero di canali, numero di punti temporali)
            n_out: numero di classi di output che il modello deve distinguere (ad esempio, target vs non-target in uno speller P300)
            alignment: stringa che determina quale metodo di allineamento tra soggetti utilizzare (None, euclidean, latent, adaptive)
            dropout: tasso di droput per la regolarizzazione del modello
        '''
        super(EEGInception, self).__init__()
        self.in_shape = in_shape
        self.n_out = n_out
        self.alignment = alignment

        n_filters = 8  # numero base di filtri convoluzionali
        n_spatial = 2  # numero di filtri spaziali per le convoluzioni che operano tra canali
        self.n_filters = n_filters
        self.n_spatial = n_spatial

        # se l'allineamento è impostato su euclidean
        if alignment == 'euclidean':
            self.euclidean = EuclideanAlignment() 
        # viene utilizzato uno strato EuclideanAlignment applicato ai dati grezzi di input
        
        # Normalizzazione dell'input
        if alignment == 'latent':
            self.latent_align0 = LatentAlignment2d(in_shape[0], affine=False)
        elif alignment == 'adaptive':
            self.abn0 = AdaptiveBatchNorm2d(in_shape[0], affine=False)
        else:
            self.bn0 = nn.BatchNorm2d(in_shape[0], affine=False)
         # a seconda del tipo di allineamento scelto, viene inizializzato uno strato di normalizzazione
        
        # Blocco Inception 1
        self.conv1a = nn.Conv2d(1, n_filters,
                                kernel_size=(1, 51), padding=(0, 25),
                                bias=True, groups=1)
        self.conv1b = nn.Conv2d(1, n_filters,
                                kernel_size=(1, 25), padding=(0, 12),
                                bias=True, groups=1)
        self.conv1c = nn.Conv2d(1, n_filters,
                                kernel_size=(1, 13), padding=(0, 6),
                                bias=True, groups=1)
        # sono tre strati Conv2d operanti sulla dimensione temporale (kernel_size=(1, X) con dimensioni di kernel diverse (51, 25, 13)
        # e padding appropriato (padding=(0, X/2) per mantenere la dimensione temporale. Ogni convoluzione produce n_filter feature map


        if alignment == 'latent':
            self.latent_align1 = LatentAlignment2d(3 * n_filters)
        elif alignment == 'adaptive':
            self.abn1 = AdaptiveBatchNorm2d(3 * n_filters)
        else:
            self.bn1 = nn.BatchNorm2d(3 * n_filters)

        # strato di allineamento /normalizzazione applicato all'output concatenato di questi tre rami. 
        # La dimensione dei canali per questi strati sarà 3*n_filters
        
        self.drop1 = nn.Dropout(dropout)
        # strato di dropout

        # Filtri spaziali
        self.conv2 = nn.Conv2d(3 * n_filters, 3 * n_spatial * n_filters,
                               kernel_size=(in_shape[-2], 1),
                               bias=False, groups=3 * n_filters)
        if alignment == 'latent':
            self.latent_align2 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn2 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.drop2 = nn.Dropout(dropout)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))

        # Inception 2
        self.conv3a = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 13), padding=(0, 6), bias=False)
        self.conv3b = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.conv3c = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 3), padding=(0, 1), bias=False)
        if alignment == 'latent':
            self.latent_align3 = LatentAlignment2d(3 * self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn3 = AdaptiveBatchNorm2d(3 * self.conv2.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(3 * self.conv2.out_channels)
        self.drop3 = nn.Dropout(dropout)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))

        # Separable Convolution 1
        self.conv4 = nn.Conv2d(3 * self.conv2.out_channels, 3 * self.conv2.out_channels,
                               kernel_size=(1, 7), padding=(0, 3), bias=False)
        if alignment == 'latent':
            self.latent_align4 = LatentAlignment2d(self.conv4.out_channels)
        elif alignment == 'adaptive':
            self.abn4 = AdaptiveBatchNorm2d(self.conv4.out_channels)
        else:
            self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.drop4 = nn.Dropout(dropout)

        # Separable Convolution 2
        self.conv5 = nn.Conv2d(self.conv4.out_channels, self.conv4.out_channels,
                               kernel_size=(1, 3), padding=(0, 1), bias=False)
        if alignment == 'latent':
            self.latent_align5 = LatentAlignment2d(self.conv5.out_channels)
        elif alignment == 'adaptive':
            self.abn5 = AdaptiveBatchNorm2d(self.conv5.out_channels)
        else:
            self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)
        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.drop5 = nn.Dropout(dropout)

        # Classifier
        self.n_features = self.conv5.out_channels * int(np.floor(in_shape[-1] / 4 / 2 / 2 / 2))
        self.fc_out = nn.Linear(self.n_features, n_out)

    def forward(self, x, sbj_trials):
        """
        Args:
             x: (batch * sbj_trials, spatial, time)
             sbj_trials: number of trials per subject
        """
        _, spatial, time = x.shape

        # Euclidean alignment
        if self.alignment == 'euclidean':
            x = self.euclidean(x, sbj_trials)

        # Input alignment
        x = x.reshape(-1, spatial, 1, time)
        if self.alignment == 'latent':
            x = self.latent_align0(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn0(x, sbj_trials)
        else:
            x = self.bn0(x)
        x = x.reshape(-1, spatial, time)

        # Create artificial image dimension
        x = x.unsqueeze(1)

        # Inception 1
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x3 = self.conv1c(x)
        x = torch.cat((x1, x2, x3), dim=1)
        if self.alignment == 'latent':
            x = self.latent_align1(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn1(x, sbj_trials)
        else:
            x = self.bn1(x)
        x = F.elu(x)
        x = self.drop1(x)

        # Spatial Filters
        x = self.conv2(x)
        if self.alignment == 'latent':
            x = self.latent_align2(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn2(x, sbj_trials)
        else:
            x = self.bn2(x)
        x = F.elu(x)
        x = self.drop2(x)
        x = self.pool1(x)

        # Inception 2
        x1 = self.conv3a(x)
        x2 = self.conv3b(x)
        x3 = self.conv3c(x)
        x = torch.cat((x1, x2, x3), dim=1)
        if self.alignment == 'latent':
            x = self.latent_align3(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn3(x, sbj_trials)
        else:
            x = self.bn3(x)
        x = F.elu(x)
        x = self.drop3(x)
        x = self.pool2(x)

        # Separable Convolution 1
        x = self.conv4(x)
        if self.alignment == 'latent':
            x = self.latent_align4(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn4(x, sbj_trials)
        else:
            x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop4(x)

        # Separable Convolution 2
        x = self.conv5(x)
        if self.alignment == 'latent':
            x = self.latent_align5(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn5(x, sbj_trials)
        else:
            x = self.bn5(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.drop5(x)

        # Classifier
        x = x.reshape(-1, self.n_features)
        x = self.fc_out(x)

        return x

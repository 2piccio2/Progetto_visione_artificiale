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
        # strato che applica filtri spaziali.
        # kernel_size=(in_shape[-2], 1) significa che la convoluzione opera su tutti i canali EEG
        # (in_shape [-2] è il numero di canali) combinandoli in nuove feature spaziali. 
        # groups=3 * n_filters indica una convoluzione a profondità separabile (depthwise separable convolution)
        # dove i filtri sono applicati indipendentemente per ciascun gruppo di canali di input. Questo
        # un modo efficente per catturare le relazioni spaziali.

        if alignment == 'latent':
            self.latent_align2 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn2 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        # strato di allineamento/normalizzazione
        self.drop2 = nn.Dropout(dropout)
        # strato di dropout
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        # strato di average pooling per ridurre la dimensione temporale

        # Blocco Inception 2
        # secondo modulo simile al primo, ma operante sulle feature ma estratte dal blocco precedente.
        # Utilizza dimensioni di kernel più piccole (13,7,3)
        self.conv3a = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 13), padding=(0, 6), bias=False)
        self.conv3b = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.conv3c = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                                kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 3 convoluzioni in parallelo
        if alignment == 'latent':
            self.latent_align3 = LatentAlignment2d(3 * self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn3 = AdaptiveBatchNorm2d(3 * self.conv2.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(3 * self.conv2.out_channels)
        # strato di allineamento/normalizzazione
        self.drop3 = nn.Dropout(dropout)
        # strato di dropout
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        # Secondo strato di average pooling

        # Convoluzioni Separabili
        # due ulteriori strati convoluzionali, probabilmente per affinare ulteriormente le feature. 
        # Anche qui, l'allineamento/normalizzazione e il pooling vengono applicati dopo ciascuno
        self.conv4 = nn.Conv2d(3 * self.conv2.out_channels, 3 * self.conv2.out_channels,
                               kernel_size=(1, 7), padding=(0, 3), bias=False)
        # strato di convoluzione con kernel_size=(1, 7).

        if alignment == 'latent':
            self.latent_align4 = LatentAlignment2d(self.conv4.out_channels)
        elif alignment == 'adaptive':
            self.abn4 = AdaptiveBatchNorm2d(self.conv4.out_channels)
        else:
            self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        # strato di normalizzazione

        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        # strato di average pooling

        self.drop4 = nn.Dropout(dropout)
        # strato di dropout
        
        self.conv5 = nn.Conv2d(self.conv4.out_channels, self.conv4.out_channels,
                               kernel_size=(1, 3), padding=(0, 1), bias=False)
        # strato di convoluzione con e kernel_size=(1, 3).

        if alignment == 'latent':
            self.latent_align5 = LatentAlignment2d(self.conv5.out_channels)
        elif alignment == 'adaptive':
            self.abn5 = AdaptiveBatchNorm2d(self.conv5.out_channels)
        else:
            self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)
         # strato di normalizzazione

        self.pool4 = nn.AvgPool2d(kernel_size=(1, 2))
        # strato di average pooling

        self.drop5 = nn.Dropout(dropout)
        # strato di dropout

        # Classificatore
        self.n_features = self.conv5.out_channels * int(np.floor(in_shape[-1] / 4 / 2 / 2 / 2))
        # calcola il numero di feature appiattite in base alle dimensioni di output finali dopo tutte
        # le convoluzioni e i pooling. L'uso di np.floor indica che si gestiscono divisioni intere.

        self.fc_out = nn.Linear(self.n_features, n_out)
        # strato fully connected finale per la classificazione.


    # metodo forward che definisce il flusso dei dati attraverso l'architettura
    def forward(self, x, sbj_trials):
        """
        Args:
             x: tensore di input, con forma (batch * sbj_trials, spatial, time)
             sbj_trials: numero di prove per ogni soggetto nel batch.
        """
        _, spatial, time = x.shape

        # Allineamento euclideo
        if self.alignment == 'euclidean':
            x = self.euclidean(x, sbj_trials)
            # viene applicato sui dati di input grezzi

        # Allineamento Input
        x = x.reshape(-1, spatial, 1, time)
        # l'input viene rimodellato per raggiungere la dimensione del canale (1) necessaria per gli strati Conv2d di PyTotch
        
        if self.alignment == 'latent':
            x = self.latent_align0(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn0(x, sbj_trials)
        else:
            x = self.bn0(x)
        # viene applicato lo strato di normalizzazione dell'input
        x = x.reshape(-1, spatial, time)
        # l'input viene rimodellato per rimuovere la dimensione del canale artificiale
    
        x = x.unsqueeze(1)
        # la dimensione del canale viene aggiunta di nuovo, questa volta per il modulo Inception che segue

        # Inception 1
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x3 = self.conv1c(x)
        # l'input viene passato attraverso i tre rami convoluzionali paralleli

        x = torch.cat((x1, x2, x3), dim=1)
        # i risultati dei tre rami vengono concatenati lungo la dimensione dei canali

        if self.alignment == 'latent':
            x = self.latent_align1(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn1(x, sbj_trials)
        else:
            x = self.bn1(x)
        # strato di allineamento/normalizzazione
        
        x = F.elu(x)
        # di attivazione F.elu (Exponetial Linear Unit)
        
        x = self.drop1(x)
        # e il dropout

        # Filtri spaziali
        x = self.conv2(x)
        # applica i filtri spaziali

        if self.alignment == 'latent':
            x = self.latent_align2(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn2(x, sbj_trials)
        else:
            x = self.bn2(x)
        # allineamento/normalizzaizione
        
        x = F.elu(x)
        # attivazione
        
        x = self.drop2(x)
        #dropout
        
        x = self.pool1(x)
        # average pooling

        # Inception 2 simile al blocco inception 1
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

        # Convoluzione Separabile 1
        x = self.conv4(x)
        # prima convoluzione separabile

        if self.alignment == 'latent':
            x = self.latent_align4(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn4(x, sbj_trials)
        else:
            x = self.bn4(x)
        # allineamento/normalizzazione

        x = F.elu(x)
        # attivazione

        x = self.pool3(x)
        # average pooling

        x = self.drop4(x)
        # dropout

        # Convoluzione separabile 2
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

        # Classificatore Finale
        x = x.reshape(-1, self.n_features)
        # le feature finali vengono appiattite

        x = self.fc_out(x)
        # le feature appiattite vengono passate allo strato lineare finale per
        # produrre i logit di classificazione. Si noti che qui non c'è un F.relu
        # prima dello strato lineare finale, il che è comune per i logit di output
        return x

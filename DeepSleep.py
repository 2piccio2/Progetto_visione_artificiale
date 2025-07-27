import torch.nn as nn
from torch.nn import functional as F
from alignment_techniques import LatentAlignment2d, AdaptiveBatchNorm2d, EuclideanAlignment


class DeepSleep(nn.Module):
    """
    DeepSleep implementation  based on
    Chambon, S., Galtier, M.N., Arnal, P.J., Wainrib, G. and Gramfort, A., 2018.
    A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series.
    IEEE Transactions on Neural Systems and Rehabilitation Engineering, 26(4), pp.758-769.
    https://ieeexplore.ieee.org/document/8307462.
    """
    '''
    La classe DeepSleep è un processo che prende il segnale EEG e cerca di capire in quale stadio del sonno si trova una persona.
    '''
    def __init__(self, in_shape, n_out, alignment='None', dropout=0.25):
        '''
        Args:
            in_shape: quanti canali (elettrodi) e quanti punti temporali ci sono nel segnale eeg in ingresso
            n_out: quanti sono gli stadi del sonno che il modello deve riconoscere (es Veglia, Sonno, etc.)
            alignment: quale tipo di metodo di allineamento utilizzare (none, euclidean, latent, adaptive)
            dropout: tecnica per evitare che il modello impari troppo a memoria e si adatti solo ai dati di addestramento
        '''
        super(DeepSleep, self).__init__()
        self.in_shape = in_shape
        self.n_out = n_out
        self.alignment = alignment

        #inizializzazione degli iperparametri e degli strati
        n_filters = 2 # numero di filtri base (moltiplicatore) per gli strati convoluzionali
        n_spatial = 8 # numero di filtri spaziali
        self.n_filters = n_filters
        self.n_spatial = n_spatial

        # se allineamento impostato su euclidean
        if alignment == 'euclidean':
            self.euclidean = EuclideanAlignment() # inizializzato uno strato EuclideanAlignment
            # viene applicato all'inizio del forward pass sui dati di input grezzi

        # Normalizzazione dell'input in base al tipo di allineamento scelto
        if alignment == 'latent':       # se allineamento impostato su latent
            self.latent_align0 = LatentAlignment2d(in_shape[0], affine=False)   # Applico LatentAlignment2d all'input, ma senza prametri affini addestrabili
        elif alignment == 'adaptive':   # se allineamento impostato su adaptive
            self.abn0 = AdaptiveBatchNorm2d(in_shape[0], affine=False)          # applico AdaptiveBatchNorm2d all'input, ma senza prametri affini addestrabili
        else:                           # se allineamento impostato su none
            self.bn0 = nn.BatchNorm2d(in_shape[0], affine=False)                # utilizza una nn.BatchNorm2d standard, senza parametri affini addestrabili


        # Filtri spaziali -> primo strato convoluzionale
        self.conv1 = nn.Conv2d(1, n_spatial,
                               kernel_size=(in_shape[0], 1),
                               bias=True)
        # convoluzione 2d che apprende filtri spaziali.
        # in_shape[0] (numero di canali eeg) come dimensione del kernel suggerisce che questo strato combina 
        # linearmente i segnali di tutti gli elettrodi per creare n_spatial mappe spaziali
        if alignment == 'latent':       # se allineamento impostato su latent
            self.latent_align1 = LatentAlignment2d(self.conv1.out_channels) # applico un altro strato di allineamento/normalizzazione in base al tipo di allineamento scelto
        elif alignment == 'adaptive':   
            self.abn1 = AdaptiveBatchNorm2d(self.conv1.out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        # Blocco 1 -> secondo strato convoluzionale
        self.conv2 = nn.Conv2d(n_spatial, n_spatial * n_filters,
                               kernel_size=(1, 51), padding=(0, 25),
                               bias=True, groups=1)
        # altro strato convoluzionale per l'estrazione delle feature temporali
        # kernel_size=(1, 51): significa che il filtro scorre nel tempo, mantenendo la dimensione spaziale.
        # padding=(0, 25): assicura che la dimensione temporale di output rimanga la stessa.

        # strato di allineamento/normalizzazione utilizzato in base al tipo di allineamento scelto
        if alignment == 'latent':
            self.latent_align2 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn2 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 16))  # strato di max pooling per ridurre la dimensione temporale del segnale

        # Blocco 2 -> terzo strato convoluzionale
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels,
                               kernel_size=(1, 51), padding=(0, 25),
                               bias=True, groups=1)
        # altro strato convoluzionale per l'estrazione delle feature temporali similare al blocco precedente

        # strato di allineamento/normalizzazione utilizzato in base al tipo di allineamento scelto
        if alignment == 'latent':
            self.latent_align3 = LatentAlignment2d(self.conv2.out_channels)
        elif alignment == 'adaptive':
            self.abn3 = AdaptiveBatchNorm2d(self.conv2.out_channels)
        else:
            self.bn3 = nn.BatchNorm2d(self.conv2.out_channels)
        
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 16))  # secondo strato di max pooling per ridurre la dimensione temporale del segnale
        self.drop1 = nn.Dropout(dropout)                # strato di dropout applicato per regolarizzazione.

        # Classificatore
        self.n_features = int(self.conv3.out_channels) * (in_shape[-1] // 16 // 16)
        # calcola il numero di feature appiattite prima dello strato fully connected.
        # Dipende dalle dimensioni di output dei precedenti strati convoluzionali e di pooling

        self.fc_out = nn.Linear(self.n_features, n_out)
        # strato fully connected finale che mappa le feature estratte al numero di classi di output (stadi del sonno).
    
    # flusso dei dati attraverso la rete    
    def forward(self, x, sbj_trials):
        """
        Args:
             x: eeg in input, formato (batch * sbj_trials, spatial, time)
             sbj_trials: number of trials per subject
        """
        _, spatial, time = x.shape

        # Allineamento euclideo
        if self.alignment == 'euclidean':
            x = self.euclidean(x, sbj_trials) 
            # Euclidean alignment applicato direttamente a x
            # Questo strato opera sulla rappresentazione spaziale-temporale del segnale

        # Preparazione dell'input per Conv2d
        x = x.reshape(-1, spatial, 1, time)
        # rimodella x per raggiungere una dimensione canale immagine (qui impostata a 1) 
        # che è necessaria per gli strati conv2d di pytorch
        
        # normalizzazione dell'input
        if self.alignment == 'latent':
            x = self.latent_align0(x, sbj_trials) # strato di allineamento che opera sulla dimensione canale immagine
        elif self.alignment == 'adaptive':
            x = self.abn0(x, sbj_trials)
        else:
            x = self.bn0(x)


        x = x.reshape(-1, spatial, time)
        # ritorna alla forma (batch * sbj, spatial, time) dopo la normalizzazione dell'input
        
        x = x.unsqueeze(1)
        # aggiunge di nuovo la dimensione canale immagine per i successivi conv2d
        
        
        # il segnale passa attraverso la serie di blocchi convoluzionali (conv1, conv2, conv3)
        # intervallati da attivazioni F.relu e strati di max pooling (pool1,pool2).
        # Dopo ogni strato convoluzionale viene applicata la rispettiva tecnica di allineamento/normalizzazione

        # Spaziale
        x = self.conv1(x)  # (batch * sbj, spatial, 1, time)
        if self.alignment == 'latent':
            x = self.latent_align1(x, sbj_trials, growing_context=growing_context) # growing_context non esiste
        elif self.alignment == 'adaptive':
            x = self.abn1(x, sbj_trials)
        else:
            x = self.bn1(x)

        # Blocco 1
        x = self.conv2(x)  # (batch * sbj, filters, spatial, time)
        if self.alignment == 'latent':
            x = self.latent_align2(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn2(x, sbj_trials)
        else:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Blocco 2
        x = self.conv3(x)
        if self.alignment == 'latent':
            x = self.latent_align3(x, sbj_trials)
        elif self.alignment == 'adaptive':
            x = self.abn3(x, sbj_trials)
        else:
            x = self.bn3(x)
        x = self.pool2(x)
        x = self.drop1(x)
        # strato di dropout applicato dopo il secondo blocco

        # Classificatore
        x = x.reshape(-1, self.n_features) 
        # appiattisco le feature estratte
        x = F.relu(x)
        # viene applicata una funzione di attivazione ReLU
        x = self.fc_out(x)
        # le feature appiattite vengono passate allo strato fully connected per produrre i logit
        # di classificazione (punteggi grezzi per ogni classe di stadio del sonno)
        return x

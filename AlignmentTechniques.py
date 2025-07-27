"""
* Copyright (C) Cogitat, Ltd.
* Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
"""
import torch
import torch.nn as nn
from torch import linalg


class LatentAlignment2d(nn.Module):
    '''
        Classe che implementa il metodo Latent Alignment
    '''
    def __init__(self, n_channels, affine=True):
        '''
        Args:
            n_channels: numero di canali delle feature su cui applicare l'allineamento
            
            affine: di tipo booleano, indica se applicare parametri di scala (weight) e bias 
            addestrabili dopo la normalizzazione. Se True, questi parametri vengono inizializzati a 
            ones e zeros rispettivamente, come nella BatchNorm standard.
        '''
        super(LatentAlignment2d, self).__init__()
        self.n_channels = n_channels
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x, sbj_trials):
        """
        Args:
            x: tensore di input. La sua forma attesa è (batch * sbj_trials, channels, spatial, time).
            Significa che il batch contiene prove di più soggetti, con sbj_trials prove per soggetto

            sbj_trials: numero di prove per soggetto all'interno del batch.
        """
        _, channels, spatial, time = x.shape

        # Rimodellazione
        x = x.reshape(-1, sbj_trials, channels, spatial, time)
        batch = x.shape[0]
        # rimodello x da (batch * sbj_trials, channels, spatial, time) a (numero_soggetti_batch, sbj_trials, channels, spatial, time).
        # passaggio fondamentale per calcolare le statistiche per ciascun soggetto individualmente, raggruppando le loro prove.

        # Standardizzazione per soggetto
        x = (x - x.mean(dim=[-4, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[-4, -2, -1], keepdim=True) + 1e-8)
        # x.mean(dim=[-4,-2,-1], keepdim=True): calcola la media delle prove per ciascun soggetto.
        # la media delle prove viene calcolata sulle prove del soggetto, su tutti i canali e le dimensioni spaziali e temporali.
        # si produce una media per soggetto (e per feature, se n_channels > 1 e le dimensioni non sono collassate).

        # x.var(dim=[-4,-2,-1], keepdim=True): calcola la varianza delle prove in modo simile.

        # (x - mean) / sqrt(var + 1e-8): standardizza (normalizzazione Z-score) utilizzando le statistiche calcolate per ciascun soggetto.
        # 1e-8 evita divisioni per zero.


        # applicazione di peso e bias (se affine=True)
        if self.affine:
            x = x * self.weight.reshape(-1, 1, 1) + self.bias.reshape(-1, 1, 1)
        # le feature standardizzate vengono scalate dai pesi (self.weight) e traslate dai bias (self.bias).
        # weight e bias sono parametri addestrabili, che consentono al modello di imparare una trasformazione ottimale dopo la standardizzazione.

        # rimodellazione inversa
        x = x.reshape(batch * sbj_trials, channels, spatial, time)
        # x viene rimodellato di nuovo alla forma originale (batch * sbj_trials, channels, spatial, time)
        # per essere passato agli strati successivi del modello.

        return x


class AdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive batchnorm implementation based on
    Li, Y., Wang, N., Shi, J., Hou, X. and Liu, J., 2018.
    Adaptive batch normalization for practical domain adaptation. Pattern Recognition, 80, pp.109-117
    https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X
    """
    '''
        Classe che imlementa una versione di Adaptive BatchNorm
    '''

    def __init__(self, n_channels, affine=True):
        '''
        Args: 
            stessi parametri di LatentAlignment
        '''

        super(AdaptiveBatchNorm2d, self).__init__()
        self.n_channels = n_channels
        self.affine = affine

        # l'unica differenza è che include uno strato di BatchNorm2d standard
        self.bn = nn.BatchNorm2d(n_channels, affine=affine)

    def forward(self, x, sbj_trials):
        """
        Args:
            x: tensore di input. La sua forma attesa è (batch * sbj_trials, channels, spatial, time).
            Significa che il batch contiene prove di più soggetti, con sbj_trials prove per soggetto

            sbj_trials: numero di prove per soggetto all'interno del batch.
        """
        _, channels, spatial, time = x.shape

        # Comportamento durante l'addestramento
        if self.training:
            x = self.bn(x)
            # si comporta come un normale Batch Norm. Le statistiche vengono calcolate su tutto il batch, che include le prove di più soggetti.
            # vengono poi utilizzate per normalizzare l'input. Le statistiche vengono aggiornate in media mobile per l'inferenza.

        # Comportamento durante l'inferenza
        else:
            
            x = x.reshape(-1, sbj_trials, channels, spatial, time)
            # rimodella x per separare le prove per soggetto, come LantentAlignment2d.
            
            x = (x - x.mean(dim=[-4, -2, -1], keepdim=True)) / torch.sqrt(x.var(dim=[-4, -2, -1], keepdim=True) + 1e-8)
            # calcola le statistiche (media e deviazione standard) per soggetto ( come in LatentAlignment2d).

            x = x.reshape(-1, channels, spatial, time)
            # standardizza x usando queste statistiche per soggetto.

            # se affine=True
            if self.affine:
                x = self.bn.weight.data.reshape(-1, 1, 1) * x + self.bn.bias.data.reshape(-1, 1, 1)
            # appplica i pesi e bias addestrati dallo stato self.bn (non vengno calcolati nuovi pesi qui).
            # questa è la differenza principale, AdaptiveBatchNorm2d utilizza i parametri affini appresi a livello 
            # globale, ma applica la normalizzazione con statistiche specifice del soggetto in inferenza.
            # rimodella inoltre x alla forma originale (batch * sbj_trials, channels, spatial, time).

        return x


class EuclideanAlignment(nn.Module):
    """
    Euclidean alignment implementation based on
    He, H. and Wu, D., 2019.
    Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach.
    IEEE Transactions on Biomedical Engineering, 67(2), pp.399-410.
    https://ieeexplore.ieee.org/abstract/document/8701679
    """

    '''
    classe che implementa il metodo di allineamento Euclideo
    '''
    def __init__(self):
        '''
        non ha parametri specifici da inizializzare, poiché non ha parametri addestrabili
        '''
        super(EuclideanAlignment, self).__init__()

    def forward(self, x, sbj_trials):
        """
        Args:
            x: tensore di input. La sua forma attesa è (batch * sbj_trials, channels, spatial, time).
            Significa che il batch contiene prove di più soggetti, con sbj_trials prove per soggetto

            sbj_trials: numero di prove per soggetto all'interno del batch.
        """

        _, spatial, time = x.shape
        
        # # rimodellazione
        x = x.reshape(-1, sbj_trials, spatial, time)
        # x viene modellato per raggruppare le prove per soggetto
        
        # Ricentraggio e Riscalatura
        x = (x - x.mean(dim=(-1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True)
        # x = (x - x.mean(dim=(-1), keepdim=True)): ricentra i segnali per elettrodo (sottrae la media temporale di ciascun elettrodo)
        # / x.std(dim=(-2, -1), keepdim=True): riscala i segnali utilizzando la deviazione standard totale per prova (su tutti gli elettrodi e il tempo),
        # anche chiamata average global field power. Questi passaggi migliorano la stabilità numerica del calcolo della covarianza.
        
        # calcolo della matrice di covarianza
        cov = torch.matmul(x, x.transpose(-2, -1)) / (x.shape[-1] - 1)
        # calcola la matrice di covarianza spaziale per ciascuna prova
        # x.transpose(-2,-1): traspone le dimensioni spaizale e temporale per il prodotto matriciale

        cond = torch.eye(cov.shape[-1], device=cov.device) * 1e-4
        # aggiunge un piccolo valore sulla diagonale (1e-4) per migliorare il condizionamento
        # della matrice ed evitare problemi numerici con matrici singolari o quasi singolari.

        cov = cov + cond
        # applica la regolarizzazione
        
        # Sbiancamento
        cov = cov.mean(dim=1, keepdim=True)
        # calcola la media delle matrici di covarianza su tutte le prove per soggetto
        
        cov = linalg.inv(linalg.cholesky(cov)).float()
        # parte cruciale, calcola la radice quadrata inversa della matrice di covarianza media.
        # linalg.cholesky(cov): esegue la decomposizione di Cholesky, che è un modo efficente per calcolare la radice quadrata
        # di una matrice definita positiva.
        # linalg.inv(): calcola l'inverso
        
        x = torch.matmul(cov, x)
        # moltiplica l'input x per la radice quadrata inversa della matrice di covarianza media.
        # questa operazione sbianca i dati rendendo le covarianze spaziali meno dipendenti dal soggetto

        # rimodellazione inversa
        x = x.reshape(-1, spatial, time)
        # x viene rimodellato alla sua forma originale
        
        return x

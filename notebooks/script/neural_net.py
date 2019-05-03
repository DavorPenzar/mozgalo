import inspect
import os
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

import score as sc
from rs_pca import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

###################################################################################################

datum = datetime.datetime.now()

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Trenirane_mreze{}:{},{}-{}'.format(str(datum.hour), str(datum.minute), str(datum.day), str(datum.month)) + '/')

os.makedirs(results_dir)


###################################################################################################
########## KLASE #################################################################################
###################################################################################################

class NeuralNet(nn.Module):
    

    def __init__(self, layers, alpha, momentum, lambda_, f = nn.Linear, act = nn.Sigmoid()):
        self.n_layers = len(layers)
        self.layers = layers
        self.alpha = alpha
        self.momentum = momentum
        self.lambda_ = lambda_
        self.name = 'L-' + str(layers) + ', a-' + str(alpha) + ', m-' + str(momentum) + ', l-' + str(lambda_)

        super(NeuralNet, self).__init__()
        if inspect.isclass(f):
            f = (self.n_layers - 1) * [f]
        for i in range(self.n_layers - 1):
            self.__setattr__('fc{0:d}'.format(i), f[i](layers[i], layers[i + 1]))
        self.act = act



    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.act(self.__getattr__('fc{0:d}'.format(i))(x))
        return x



    def train(self, criterion = nn.BCELoss(), optimizer = None, n_epochs = 8000, batch_size = 10, eps = 1.0e-7):
        print('\n--------------------------------------------------------------\n')
        print('Trening neuronske mreze sa parametrima alpha: {}, momentum: {}, lambda: {}'.format(alpha, momentum, lambda_))

        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=self.alpha, momentum=self.momentum, weight_decay=self.lambda_)

        losses = []
        prekini_trening = False

        for epoch in range(n_epochs):
            for i in range(0, len(X_train), batch_size):
                podaci = X_train[i:(i+batch_size), :]
                labele = y_train[i:(i+batch_size)].reshape(len(y_train[i:(i+batch_size)]), 1)

                podaci = Variable(podaci, requires_grad=True).to(device)
                labele = Variable(labele, requires_grad=False).to(device)

                # obrisi povijest operacija koje su se dogodile
                optimizer.zero_grad()

                # loss sadrzi sve operacije unazad (npr. input pa cijeli forward pa criterion) i na njemu
                # mozemo napraviti backpropagation preko funkcije backward
                # nekad se zna dogoditi 'RuntimeError: reduce failed to synchronize: device-side assert triggered'
                # pa ovime izbjegavamo taj problem jer nemam pojma zasto bi se to dogodilo
                try:
                    outputs = self.forward(podaci)
                    loss = criterion(outputs, labele)
                    loss.backward()
                except RuntimeError:
                    return -1

                # updateaj novonastale weight-ove
                optimizer.step()

                # spremimo sve vrijednosti loss-eva
                if(i + batch_size >= len(X_train)):
                    greska = loss.data.cpu().numpy()
                    losses.append(greska)
                    
                    if(epoch >= 5):
                        prekini_trening = (max(abs(greska - l) for l in losses[-4:]) < eps)
            
            if(prekini_trening):
                print("Trening konvergira u epohi {} sa loss-om: {}".format(epoch, losses[-1]))
                break
            
            if not (epoch % 25):
                print("Epoch {} - loss: {}".format(epoch, losses[-1]))
            
            
        return losses

###################################################################################################



###################################################################################################
########## FUNKCIJE ###############################################################################
###################################################################################################

def normalize_train(X):
    meanovi = []
    stdovi = []

    for k in range(X.shape[1]):
        meanovi += [np.mean(X[:,k])]
        stdovi += [np.std(X[:,k])]

        X[:,k] = ( X[:,k] - meanovi[-1] )/stdovi[-1]

    return meanovi, stdovi

def normalize_others(X, meanovi, stdovi):
    if(len(meanovi) != X.shape[1]):
        raise RuntimeError('ovaj skup nema isti broj stupaca kao i trening skup! Sredi to')

    for k in range(X.shape[1]):
        X[:,k] = ( X[:,k] - meanovi[k] )/stdovi[k]    
  

def acc_f1(neural_net, X, y, threshold=0.5):
    '''
        Funkcija koja racuna accuracy i F1-score za podatke X provedene kroz mrezu neural_net,
        i njihove labele y

        Povratni argument je tuple te dvije vrijednosti
    '''
    with torch.no_grad():
        tp, fp, tn, fn = cirkus_kratica(neural_net, X, y, threshold)
        # print('\nTP: {}, FP: {}, TN: {}, FN: {}'.format(tp,fp,tn,fn))

        precision = (tp/(tp+fp)) if (tp+fp) else 0.0
        recall = (tp/(tp+fn)) if (tp+fn) else 0.0
    
        accuracy = (tp+tn)/(tp + fp + tn + fn) 
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0

        # printaj i prvih 5 primjera, cisto da vidimo da nisu sve iste vrijednosti
        print('Neki od rezultata:')
        print(neural_net.forward(X[:5, :].to(device)))
        print('\n')

        return (accuracy, f1, precision, recall)


def sortiraj_mreze(neural_nets_names, eval_of_nets):
    '''
        Funkcija koja sortira listu neural_nets_names silazno na osnovu eval_of_nets
    '''
    with torch.no_grad():
        for i in range(len(eval_of_nets) - 1):
            for j in range(len(eval_of_nets) - i - 1):
                if(eval_of_nets[j] < eval_of_nets[j+1]):
                    eval_of_nets[j], eval_of_nets[j+1] =  eval_of_nets[j+1], eval_of_nets[j]
                    neural_nets_names[j], neural_nets_names[j+1] =  neural_nets_names[j+1], neural_nets_names[j]

                
def cirkus_kratica(neural_net, X, y, threshold=0.5):
    '''
        Funkcija koja vraca TP, FP, TN, FN vrijednosti za podatke X, provedene kroz mrezu neural_net,
        i njihove labele y 
    '''
    predicted = neural_net.forward(X.to(device)).cpu().data.numpy().astype(np.float16).ravel()
    y = y.data.numpy().astype(np.int8).ravel()
    
    return(
        ((predicted > threshold) & (y == 1)).sum(), 
        ((predicted > threshold) & (y == 0)).sum(),
        ((predicted <= threshold) & (y == 0)).sum(),
        ((predicted <= threshold) & (y == 1)).sum()
    )


def smanji_pca(dataset1, dataset2, broj_stupaca_za_sacuvati = 5):
    '''
        Funkcija koja u dataset-u brise sve stupce nastale rs-pca postupkom ciji je header
        strogo veci od broj_stupaca_za_sacuvati. Po default-u je to broj 5
    '''

    ## odredivanje broja pca stupaca
    if(dataset1.shape[1] != dataset2.shape[1]):
        raise RuntimeError('Trening i validajski skup imaju razlicit broj feature-a! Popravi to.')

    broj_pca_stupaca = 0

    for stupac in dataset1.columns.values:
        if(isinstance(stupac, int)):
            broj_pca_stupaca += 1

    stupci_za_izbaciti = []
    for i in range(broj_stupaca_za_sacuvati, broj_pca_stupaca, 1):
        stupci_za_izbaciti += [i]

    dataset1.drop(columns=stupci_za_izbaciti, inplace=True)
    dataset2.drop(columns=stupci_za_izbaciti, inplace=True)

###################################################################################################














###################################################################################################
#####    MAIN      ################################################################################
###################################################################################################



###################
###### TRENING SKUP
###################
path_trening = '../../data/mocni_trening.pkl'
trening = pd.read_pickle(path_trening)

path_validacijski = '../../data/mocni_validacijski.pkl'
validacijski = pd.read_pickle(path_validacijski)


## definiranje stupaca za izbacivanje iz tablice
stupci_za_dropat = [
    'KLIJENT_ID', ### nema smisla da je unutra
    'OZNAKA_PARTIJE', ### nema smisla da je unutra
    'VALUTA', ### ovih 5 ispod je izbaceno jer su kategorijski feature-i
    'VRSTA_KLIJENTA',
    'PROIZVOD',
    'VRSTA_PROIZVODA',
    'TIP_KAMATE',
    'GDP per capita, current prices\n (U.S. dollars per capita)', ### ovi ispod su ekonomski pokazatelji koje smo eminirali zbog koreliranosti
    'GDP, current prices (Purchasing power parity; billions of international dollars)',
    'GDP per capita, current prices (Purchasing power parity; international dollars per capita)',
    'Implied PPP conversion rate (National currency per international dollar)',
    'Inflation rate, end of period consumer prices (Annual percent change)',
    'Population (Millions of people)',
    'Current account balance\nU.S. dollars (Billions of U.S. dollars)',
    'Net lending/borrowing (also referred as overall balance) (% of GDP)',
    'Primary net lending/borrowing (also referred as primary balance) (% of GDP)',
    'Expenditure (% of GDP)',
    'Gross debt position (% of GDP)'
    ]

print(trening.shape, validacijski.shape)

trening.drop(columns=stupci_za_dropat, inplace=True)
validacijski.drop(columns=stupci_za_dropat, inplace=True)

## sadrzi samo neke(od njih broj_pca_stupaca) stupaca nastalih rs-pca-om
smanji_pca(trening, validacijski, broj_stupaca_za_sacuvati=4)

print(trening.shape, validacijski.shape)

## pogledajmo skupove za svaki slucaj
# print(trening.head())
# print(list(trening.columns.values))
# print(validacijski.head())
# print(list(validacijski.columns.values))

    
## pretvori sve u numpy koji ce se splitat na trening i validacijski skup, a kasnije ce se oni cast-ati u tensor
X_train = np.array(trening.drop(columns=['PRIJEVREMENI_RASKID']).values, dtype=np.float32)
X_val = np.array(validacijski.drop(columns=['PRIJEVREMENI_RASKID']).values, dtype=np.float32)

y_train = np.array(trening['PRIJEVREMENI_RASKID'], dtype=np.float32)
y_val = np.array(validacijski['PRIJEVREMENI_RASKID'], dtype=np.float32)
del trening
del validacijski


## pretvorba iz numpy-a u torch
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)

############################



#######################################################################
######### TRENING NEURONSKE MREZE    ##################################
#######################################################################


# run-aj na GPU-u ako mozes
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Pokrecem rad na ' + str(device))

###########################
### SLJEDECE PARAMETRE 
### TREBA RUCNO POSTAVLJATI 
### OVDJE U KODU
###########################

# kombinacije layer-a
velicina_ulaza = X_train.shape[1]

layers_combination = [
    [velicina_ulaza, 10, 10, 1]
    ]

# parametri
alphas = [0.5]
momentums = [0.75]
lambdas = [0.5] # nijednom dosad nije uspjelo treniranje sa pozitivnim vrijednostima od lambda


# KOLIKO ZELIMO najboljih mreza spremiti
br_najboljih_mreza = 5

###########################

neural_nets_names  = []
eval_of_nets = []
nenatrenirane_mreze = []


for layers in layers_combination:
    for alpha in alphas:
        for momentum in momentums:
            for lambda_ in lambdas:
                neural_net = NeuralNet(layers, alpha, momentum, lambda_).to(device)
                
                # trening
                net_loss = neural_net.train(batch_size=80000)

                # ukoliko se dogodilo neuspjelo treniranje zbog RuntimeError-a
                if(net_loss == -1): 
                    print('Treniranje ove mreze nije uspjelo!')
                    # spremi imena tih mreza koje se nisu uspjele natrenirati zbog ovog error-a
                    nenatrenirane_mreze += [neural_net.name]
                    del neural_net
                    continue


                # racunanje score-a
                acc_f1_score = acc_f1(neural_net, X_val, y_val)
                acc_str = "{:.3f}".format(acc_f1_score[0])
                f1_str  = "{:.3f}".format(acc_f1_score[1])
                prec_str  = "{:.3f}".format(acc_f1_score[2])
                recall_str  = "{:.3f}".format(acc_f1_score[3])
                rez = 'Accuracy: ' + acc_str + ', F1: ' + f1_str + ', Precision: ' + prec_str + ', Recall: ' + recall_str

                print(rez)

                # spremanje rezultata i imena mreze radi sortiranja koji ce doci kasnije
                eval_of_nets += [acc_f1_score]
                neural_nets_names += [neural_net.name]

                # spremi mrezu i sliku grafa loss-eva tijekom iteracija
                torch.save(neural_net.state_dict(), results_dir + neural_net.name + '.pt')

                fig, ax = plt.subplots()
                ax.set_title(rez)
                ax.plot(net_loss)

                fig.savefig(results_dir + neural_net.name + '.png')
                plt.close(fig)

                #### fig.close ili tak nest

                # ukoliko vec imamo vise od br_najboljih_mreza(tj. tocno 6 njih),  obrisi najslabiju mrezu i njezinu sliku grafa
                if(len(neural_nets_names) > br_najboljih_mreza):
                    # sortirajmo "mreze" po efikasnosti
                    sortiraj_mreze(neural_nets_names, eval_of_nets)

                    # obrisi potrebne stvari od najgore mreze
                    os.remove(results_dir + neural_nets_names[-1] + '.pt')
                    os.remove(results_dir + neural_nets_names[-1] + '.png')

                    eval_of_nets = eval_of_nets[:-1]
                    neural_nets_names = neural_nets_names[:-1]


                # uklananje objekta
                del neural_net

##############################################################################




print('\nNajboljih 5 rezultata:')
for i in range(min([br_najboljih_mreza, len(neural_nets_names)])):
    print('Accuracy: {:.2f}, F1_score: {:.2f}'.format(eval_of_nets[i][0], eval_of_nets[i][1]))


print('\nMreze na kojima nije uspio trening zbog RuntimeError-a: ')
for ime_mreze in nenatrenirane_mreze:
    print(ime_mreze)
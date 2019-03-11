# -*- coding: utf-8 -*-
# main-mrf.py
# author : Antoine Passemiers

from deepcp.prot.contact_map import ContactMap
from deepcp.prot.evaluation import Evaluation
from deepcp.prot.parsers import *
from deepcp.prot.data_manager import DataManager
from deepcp.prot.profiles import PSSMSampler, HMMSampler, MixtureSampler, NeighbourhoodSampler
from deepcp.prot.utils import *
from deepcp.prot.sequence import Sequence, N_AA_SYMBOLS, N_STATES

from functools import reduce
import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')


DATA_PATH = '../data'
TEMP_PATH = '../out/temp'

min_aa_separation = 6
contact_threshold = 8
identity_threshold = 0.8
max_gap_fraction = 0.9

lambda_j = 0.005
lambda_h = 0.01
label_smoothing = False
batch_size = 64
num_epochs = 10
max_num_steps = 5000
learning_rate = 0.0001
momentum_decay_first_order = 0.5
momentum_decay_second_order = 0.999


class LinearNd(torch.nn.Module):

    def __init__(self, in_shape, out_shape, bias=True):
        super(LinearNd, self).__init__()
        self.in_shape, self.in_size = self.check_shape_and_size(in_shape)
        self.out_shape, self.out_size = self.check_shape_and_size(out_shape)
        self.linear = torch.nn.Linear(self.in_size, self.out_size, bias=bias)
        #self.reset_parameters()
    
    def reset_parameters(self):
        std = np.sqrt(2. / self.in_size)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=std)
    
    def check_shape_and_size(self, shape):
        if isinstance(shape, int):
            return (shape,), shape
        elif isinstance(shape, tuple):
            return shape, reduce((lambda x, y: x * y), shape)
        else:
            return (int(shape),), int(shape)
    
    def forward(self, X):
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        out = self.linear(X)
        out_shape = tuple([batch_size] + list(self.out_shape))
        return out.view(*out_shape)

    def get_weights(self):
        shape = tuple(list(self.in_shape) + list(self.out_shape))
        return self.linear.weight.view(*shape)


class Discriminator(torch.nn.Module):

    def __init__(self, L, reg='L2'):
        super(Discriminator, self).__init__()
        self.L = L
        self.J = LinearNd((self.L, N_STATES, self.L, N_STATES), 1, bias=False)
        self.H = LinearNd((self.L, N_STATES), 1, bias=True)
        self.activation = torch.nn.Sigmoid()
        self.reg = reg
        self.dropout = torch.nn.Dropout(0.2)
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(N_STATES**2, 4, 3, padding=1, stride=1),
            torch.nn.Tanh())

    def forward(self, X):
        n_samples = X.size()[0]
        reg_j, reg_h = self.regularization()
        XX_prime = torch.einsum('biq,bjr->biqjr', [X, X])
        #XX_prime = XX_prime.view(n_samples, N_STATES ** 2, self.L, self.L)
        #XX_prime = self.conv2d(XX_prime)

        j_sum = self.J(XX_prime).squeeze(1)
        h_sum = self.H(X).squeeze(1)
        energy = j_sum + h_sum

        out = self.activation(energy)
        return out, energy, reg_j, reg_h
    
    def regularization(self):
        if self.reg == 'L2':
            reg_j = (self.J.get_weights() ** 2).sum()
            reg_h = (self.H.get_weights() ** 2).sum()
        else:
            reg_j = torch.sum(torch.abs(self.J.get_weights()))
            reg_h = torch.sum(torch.abs(self.H.get_weights()))
        return reg_j, reg_h
    
    def get_weights(self):
        J = np.squeeze(self.J.get_weights().data.numpy())
        H = np.squeeze(self.H.get_weights().data.numpy())
        J_full = np.transpose(J, (0, 2, 1, 3))
        return J_full, H


class Generator(torch.nn.Module):

    def __init__(self, L):
        super(Generator, self).__init__()
        self.L = L
        self.latent_dim = 64
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 4096),
            torch.nn.Tanh(),
            LinearNd(4096, (self.L, N_STATES)))
        self.activation = torch.nn.Softmax(dim=2)
    
    def generate(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        z = Variable(torch.Tensor(noise))
        out = self.linear(z)
        out = self.activation(out)
        return out


def gandca(msa, msa_weights, target_cmap=None):

    indices = [i for i, sequence in enumerate(msa) if sequence.gap_fraction() < max_gap_fraction]
    msa = [msa[i] for i in indices]
    #msa_weights = msa_weights[indices]
    msa_weights = np.ones(len(indices))


    alignment = np.asarray([sequence.to_array(states=True) for sequence in msa], dtype=np.uint8)
    
    all_weights = np.copy(msa_weights)
    print(all_weights, type(all_weights))
    all_weights *= (float(len(all_weights)) / all_weights.sum())
    
    sampler = MixtureSampler(msa, all_weights)
    neighbourhood_sampler = NeighbourhoodSampler(msa, all_weights)


    L = len(msa[0])
    G = Generator(L)
    D = Discriminator(L, reg='L2')
    
    # Initialize optimizers
    b1 = momentum_decay_first_order
    b2 = momentum_decay_second_order

    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(b1, b2))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    # Initialize adversarial loss
    criterion = torch.nn.BCELoss()

    step = 0
    real_energies, fake_energies, random_energies, crandom_energies, neigh_energies = list(), list(), list(), list(), list()
    ppvs, d_training_loss, g_training_loss = list(), list(), list()
    for epoch in range(num_epochs):
        all_indices = np.arange(len(msa))
        random.shuffle(all_indices)

        shuffled_msa = [msa[i] for i in all_indices]
        for batch_id in range(len(shuffled_msa) // batch_size):
            indices = all_indices[batch_id*batch_size:(batch_id+1)*batch_size]
            Y = np.asarray([shuffled_msa[i].to_array(states=True, one_hot_encoded=True) for i in indices])
            Y = Variable(torch.Tensor(Y))

            weights = Variable(torch.Tensor(all_weights[indices]))

            # Ground truth labels
            print('\tGenerating labels...')
            if label_smoothing: # One-sided label smoothing
                valid = Variable(torch.FloatTensor(batch_size).uniform_(0.7, 1.2), requires_grad=False)
            else:
                valid = Variable(torch.Tensor(batch_size).fill_(1.), requires_grad=False)
            fake = Variable(torch.Tensor(batch_size).fill_(0.), requires_grad=False)

            """
            # Train generator G
            print('\tTraining G...')
            g_optimizer.zero_grad()
            generated = G.generate(batch_size)
            d_predictions, _, _, _ = D.forward(generated)
            g_loss = criterion(d_predictions, valid)
            g_loss.backward()
            g_optimizer.step()
            """
            generated = Variable(torch.Tensor(sampler.sample(batch_size)))
            
            """
            for i in range(batch_size):
                sequence = Sequence(generated[i].data.numpy().argmax(axis=1))
                print(sequence)
            """

            # Train discriminator D
            print('\tTraining D...')
            d_optimizer.zero_grad()
            real_classified, real_energy, reg_j, reg_h = D.forward(Y)
            tp = np.mean(real_classified.data.numpy() >= 0.5)
            weighted_criterion = torch.nn.BCELoss(weight=weights)
            real_loss = weighted_criterion(real_classified, valid) + lambda_j * reg_j + lambda_h * reg_h

            fake_classified, fake_energy, reg_j, reg_h = D.forward(generated.detach())
            tn = np.mean(fake_classified.data.numpy() < 0.5)
            fake_loss = criterion(fake_classified, fake) + lambda_j * reg_j + lambda_h * reg_h
            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            d_optimizer.step()

            # Generate random sequences
            Y = np.asarray([Sequence.random(L).to_array(states=True, one_hot_encoded=True) for i in indices])
            Y = Variable(torch.Tensor(Y))
            _, random_energy, _, _ = D.forward(Y)
            Y = np.random.rand(batch_size, L, N_STATES)
            Y /= Y.sum(axis=2)[:, :, None]
            Y = Variable(torch.Tensor(Y))
            _, crandom_energy, _, _ = D.forward(Y)
            Y = neighbourhood_sampler.sample(batch_size)
            Y = Variable(torch.Tensor(Y))
            _, neigh_energy, _, _ = D.forward(Y)


            if target_cmap:
                # Saving evaluation metrics and plots
                # J = G.get_weights()
                J, H = D.get_weights()
                pred_cmap = ContactMap(couplings_to_cmap(J))
                entry = Evaluation().add('-', '-', pred_cmap, target_cmap, min_aa_separation)
                print('\n[Epoch %d/%d] [D loss: %f]' % (epoch+1, num_epochs, d_loss.item()))
                print('Discriminator - TP: %f - TN: %f' % (tp, tn))
                print('PPV: %f' % entry['PPV'])
                ppvs.append(entry['PPV'])

                if step % 10 == 0:
                    print('\tSaving plots...')
                    plt.imshow(pred_cmap)
                    plt.title('PPV: %f' % entry['PPV'])
                    plt.savefig('generated/training/%i.png' % step)
                    plt.clf()

            if step % 10 == 0:
                plt.plot(real_energies, label='Real sequences')
                plt.plot(fake_energies, label='Fake sequences')
                plt.plot(random_energies, label='Random sequences (d)')
                plt.plot(crandom_energies, label='Random sequences (c)')
                plt.plot(neigh_energies, label='Neighbour sequences')
                plt.title('')
                plt.xlabel('Step')
                plt.ylabel('Average energy function')
                plt.legend()
                plt.savefig('generated/energies.png')
                plt.clf()
                
                chart(d_training_loss, ppvs)

            # g_training_loss.append(g_loss.item())
            d_training_loss.append(d_loss.item())
            real_energies.append(float(real_energy.mean()))
            fake_energies.append(float(fake_energy.mean()))
            random_energies.append(float(random_energy.mean()))
            crandom_energies.append(float(crandom_energy.mean()))
            neigh_energies.append(float(neigh_energy.mean()))
            step += 1
                
            if step >= max_num_steps:
                return D.get_weights()
            
    return D.get_weights()

def chart(d_loss, ppvs) :
    fig = plt.figure()
    host = fig.add_subplot(111)
    par1 = host.twinx()
    #par2 = host.twinx()
    #host.set_xlim(0, 2)
    #host.set_ylim(0, 2)
    #par1.set_ylim(0, 4)
    #par2.set_ylim(1, 65)
    host.set_xlabel('Steps')
    host.set_ylabel('Discriminator loss')
    par1.set_ylabel('Best-L PPV')
    #par2.set_ylabel('')

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    #color3 = plt.cm.viridis(.9)

    p1, = host.plot(d_loss, color=color1,label='Discriminator loss')
    p2, = par1.plot(ppvs, color=color2, label='Best-L PPV')
    #p3, = par2.plot([0, 1, 2], [50, 30, 15], color=color3, label="Velocity")

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    # par2.spines['right'].set_position(('outward', 60))      
    # no x-ticks                 
    # par2.xaxis.set_ticks([])
    # Sometimes handy, same for xaxis
    #par2.yaxis.set_ticks_position('right')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    # par2.yaxis.label.set_color(p3.get_color())

    plt.savefig('generated/loss_dca.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    
    evaluation = Evaluation()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)

    # Saves short-range PPV, medium-range PPV, long-range PPV, MCC, accuracy, etc.
    print('Predict on validation set...')
    for feature_set in data_manager.proteins(dataset='debug_dca'):
        L = len(feature_set.msa[0])
        seq_name = feature_set.prot_name
        target_cmap = ContactMap.from_distances(feature_set.distances, T=contact_threshold)

        for idx, feature_name in [(1, 'GaussDCA'), (2, 'plmDCA')]:

            try:
                pred_cmap = ContactMap(feature_set.features['2-dim']['values'][idx])

                entry = evaluation.add(feature_name, seq_name, pred_cmap, target_cmap, min_aa_separation)
                print('PPV of %s: %f' % (feature_name, entry['PPV']))

                plt.imshow(pred_cmap.cmap)
                plt.title('PPV: %f' % entry['PPV'])
                plt.savefig('generated/%s.png' % feature_name)
                plt.clf()
            except:
                pass


        J = gandca(feature_set.msa, feature_set.msa_weights, target_cmap)
        pred_cmap = ContactMap(couplings_to_cmap(J))

        entry = evaluation.add('gan', seq_name, pred_cmap, target_cmap, min_aa_separation)
        print("PPV for protein %s: %f" % (seq_name, entry["PPV"]))

        break

    for feature_name in ['PSICOV', 'GaussDCA', 'plmDCA']:
        best_l = evaluation.get(feature_name, 'PPV')
        sh = evaluation.get(feature_name, 'PPV-short')
        med = evaluation.get(feature_name, 'PPV-medium')
        lg = evaluation.get(feature_name, 'PPV-long')
        print('\n                                   All         Short       Medium      Long')
        print('Average best-L PPV for %s: %0.3f       %0.3f       %0.3f       %0.3f' % \
            (feature_name.ljust(10), best_l, sh, med, lg))

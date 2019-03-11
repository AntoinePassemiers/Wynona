import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
#from cgancon.dca.gan import *
from gan import *


if __name__ == '__main__':

    L = 80
    batch_size = 32
    sequences = [np.random.randint(0, 21, size=L) for i in range(batch_size)]
    labels = np.random.randint(0, 2, size=batch_size)
    D = Discriminator(L)
    out = D.forward(sequences)
    print(out)
    for i in range(100):
        print(i)
        out = D.forward(sequences)
        D.backward(out, labels, sequences)
    out = D.forward(sequences)
    print(out)
    print('Finished')
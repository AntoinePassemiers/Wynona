import os
import numpy as np
import pandas as pd
import pconsc4


def parse_gdca(filepath, delimiter=' ', target_col=2, sequence_length=None):
    with open(filepath, "r") as f:
        lines = f.readlines()
        try:
            if sequence_length is None:
                column_names = ['i', 'j', 'target_col']
                dataframe = pd.read_csv(filepath, sep=delimiter, skip_blank_lines=True,
                    header=None, names=column_names, index_col=False)
                sequence_length = np.asarray(dataframe[["i", "j"]]).max()
        except ValueError:
            return None
        data = np.full((sequence_length, sequence_length), np.nan, dtype=np.double)
        np.fill_diagonal(data, 0)
        for line in lines:
            elements = line.rstrip("\r\n").split(delimiter)
            i, j, k = int(elements[0]) - 1, int(elements[1]) - 1, float(elements[target_col])
            data[i, j] = data[j, i] = k
        if np.isnan(data).any():
            # sequence_length is wrong or the input file has missing pairs
            warnings.warn("Warning: Pairs of residues are missing from the contacts text file")
            warnings.warn("Number of missing pairs: %i " % np.isnan(data).sum())
        return data



model = pconsc4.get_pconsc4()


for dataset in ['benchmark_set', 'benchmark_set_cameo', 'benchmark_set_casp11', 'benchmark_set_membrane', 'PSICOV150', 'training_set']:
    for folder in os.listdir(dataset):
        folder = os.path.join(dataset, folder)
        target_filepath = os.path.join(folder, 'pconsc4.out')
        if os.path.isfile(os.path.join(folder, 'alignment.a3m')):
            alignment_filepath = os.path.join(folder, 'alignment.a3m')
            gaussdca = parse_gdca(os.path.join(folder, 'gdca.out'), delimiter=' ')
        else:
            alignment_filepath = os.path.join(folder, 'sequence.fa.blits4.trimmed')
            gaussdca = parse_gdca(os.path.join(folder, 'fnr.gaussdca'), delimiter=' ')
        print(alignment_filepath)
        pred = pconsc4.predict(model, alignment_filepath, gaussdca)
        cmap = pred['cmap']
        if not os.path.isfile(target_filepath):
            with open(target_filepath, 'w') as f:
                for i in range(cmap.shape[0]):
                    for j in range(i):
                        f.write('%i,%i,%f\n' % (i+1, j+1, cmap[i, j]))

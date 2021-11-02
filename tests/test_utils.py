from chimaera.utils import *
import cooler

from urllib import request
import gzip, shutil
import os

cooler_link = "https://osf.io/hr4xb/download"
fasta_link = "https://osf.io/xursa/download"

INPUT_COOLER = './tests/data/HFF_chr2-chr10_hg38_4DNFIP5EUOFX.mapq_30.2048.cool'
INPUT_FASTA = './tests/data/chr2-chr10.hg38.fa'

if not os.path.isfile(INPUT_COOLER) or not os.path.isfile(INPUT_FASTA):
    request.urlretrieve(cooler_link, INPUT_COOLER)
    request.urlretrieve(fasta_link, INPUT_FASTA + '.gz')
    with gzip.open(INPUT_FASTA + '.gz', 'rb') as f_in, open(INPUT_FASTA, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def test_scale_y(request, tmpdir):
    # TODO
    # y =
    pass

def test_split_data(request, tmpdir):

    clr = cooler.Cooler(INPUT_COOLER)
    bin_table = clr.bins()[:]
    data_size = len(bin_table)

    assert np.all(split_data(bin_table, method='test', params={})[0] == np.arange(data_size))

    # Equal-sized split, random mode:
    assert split_data(bin_table, method='random', params={'val_split': 0.5})[0].shape[0] == data_size // 2
    assert split_data(bin_table, method='random', params={'val_split': 0.5})[1].shape[0] == data_size // 2

    # Non-equal-sized split, random mode:
    assert split_data(bin_table, method='random', params={'val_split': 0.75})[1].shape[0] == 3 * data_size // 4
    assert split_data(bin_table, method='random', params={'val_split': 0.25})[1].shape[0] == 1 * data_size // 4

    # Equal-sized split, first and last modes:
    assert split_data(bin_table, method='first', params={'val_split': 0.5})[1].shape[0] == data_size // 2
    assert split_data(bin_table, method='last', params={'val_split': 0.5})[1].shape[0] == data_size // 2



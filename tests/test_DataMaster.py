import chimaera
import pandas as pd

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

def test_split_data(request, tmpdir):

    viewframe = pd.DataFrame({
        'chrom' : ['chr2', 'chr2', 'chr10'],
        'start' : [0, 5_000_000, 10_00_000],
        'end'   : [2_500_000, 7_500_000, 15_000_000]
                             })

    data = chimaera.DataMaster(INPUT_COOLER, INPUT_FASTA,
                               2048*100, # DNA length
                               2048*100, # Hi-C length
                               viewframe=viewframe,
                               nthreads_snipper=10)

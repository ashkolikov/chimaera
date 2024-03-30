import numpy as np
import pandas as pd
from scipy import stats

from .ism import SeqMatrix, MutGenerator
from . import plot_utils
from . import data_utils


class GeneSampler(SeqMatrix):
    '''Slices genes (or other specified sequences) by coordinates in genome'''
    def __init__(self, data, gene_table, target_length):
        self.gene_table = gene_table
        self.target_length = target_length
        self.data = data

    def spawn(self, ind=None, rc=False):
        if ind is None:
            ind = np.random.choice(len(self.gene_table))
        row = self.gene_table.iloc[ind]
        start = int(row.start)
        end = int(row.end)
        if self.target_length is not None:
            real_length = row.end - row.start
            if self.target_length < real_length:
                raise ValueError('Gene is longer than target length')
            '''else:
                flanks_length = self.target_length - real_length
                start = start - flanks_length // 2
                end = end + flanks_length - flanks_length // 2'''
        gene = self.data.get_dna(f'{row.chrom}:{start}-{end}', seq=False)[0]
        if (row.strand == '-' and not rc) or (row.strand == '+' and rc):
            gene = np.flip(gene)
        return gene

    def __len__(self):
        return int(self.target_length)

def outer_intervals(intervals, mn, mx):
    '''Makes outer intervals for specified intervals'''
    next_start = mn
    for x in intervals:
        if next_start < x[0]:
            yield next_start,x[0]
            next_start = x[1]
        elif next_start < x[1]:
            next_start = x[1]
    if next_start < mx:
        yield next_start, mx

def select_genes(
    table,
    min_len=0,
    max_len=np.inf,
    min_score=-np.inf,
    max_score=np.inf
    ):
    '''Selects genes with specified length and scores (may be expression or \
smth else)'''
    lens = (table.end - table.start)
    if hasattr(table, 'score'):
        good_genes = table.loc[((min_len <= lens) & (lens <= max_len)) &
                            ((min_score <= table.score) & (table.score <= max_score))]
    else:
        good_genes = table.loc[((min_len <= lens) & (lens <= max_len))]
    return good_genes

def select_igrs(table):
    '''Makes table of intergenic regions'''
    igr_table = {'chrom':[], 'start':[], 'end':[]}
    for chrom in set(table.chrom):
        chrom_table = table.loc[table.chrom == chrom]
        # using IGRs only between first and last genes:
        mn, mx = chrom_table.end.min(), chrom_table.start.max()
        igrs = list(outer_intervals(zip(chrom_table.start, chrom_table.end), mn, mx))
        igr_table['chrom'] += [chrom]*len(igrs)
        igr_table['start'] += [i[0] for i in igrs]
        igr_table['end'] += [i[1] for i in igrs]
    return pd.DataFrame(igr_table)

def gene_composition(Model,
                     table,
                     max_len='auto',
                     min_len='auto',
                     min_score=-np.inf,
                     max_score=np.inf,
                     long_spacer_length='auto',
                     between_genes='auto',
                     upstream=0,
                     downstream=0,
                     n_replicates=64,
                     composition='>>><<<_>>><<<',
                     experiment_index=0,
                     edge_policy='empty'
                     ):
    '''Predicts multiple random compositions of genes and intergenic regions in \
specified orientation'''
    gene_table = table.copy()
    plus_strand = gene_table.strand=='+'
    minus_strand = gene_table.strand=='-'
    gene_table.loc[plus_strand, 'start'] = np.maximum(gene_table.start[plus_strand] + upstream, 0)
    gene_table.loc[minus_strand, 'end'] = gene_table.end[minus_strand] - upstream
    gene_table.loc[plus_strand, 'end'] = gene_table.end[plus_strand] + downstream
    gene_table.loc[minus_strand, 'start'] = np.maximum(gene_table.start[minus_strand] - downstream, 0)

    if max_len == 'auto' or min_len == 'auto':
        starts = np.array(gene_table.start)
        ends = np.array(gene_table.end)
        median_len = np.median(ends-starts)
        if max_len == 'auto':
            max_len = int(median_len*1.2)
        if min_len == 'auto':
            min_len = int(median_len*0.8)

    genes = select_genes(gene_table, min_len=min_len, max_len=max_len,
                         min_score=min_score, max_score=max_score)
    igrs = select_igrs(gene_table)
    igr_gen = data_utils.ChimericDNALoader(Model.data, igrs, Model.data.dna_len,
                                size=n_replicates, edge_policy=edge_policy)
    gene_gen = GeneSampler(Model.data, genes, max_len)
    if between_genes == 'auto':
        between_genes = int(np.median(igrs.end-igrs.start))

    gen = MutGenerator(igr_gen,
                       gene_gen,
                       batch_size=Model.batch_size,
                       composition=composition,
                       strategy='one_to_all',
                       long_spacer_length=long_spacer_length,
                       between_insertions=between_genes,
                       offset=Model.data.offset,
                       return_wt_too=True # predictions will be normalized by predictions from seqs without genes
                       )
    y = Model.predict(gen, strand='one')
    half_1, half_2 = y[:n_replicates//2], y[n_replicates//2:]
    pearson = stats.pearsonr(half_1.mean(axis=0).flat, half_2.mean(axis=0).flat)
    if pearson.pvalue < 0.05:
        verdict = f' correlate significantly with Pearson r={pearson.statistic:.2f}'
        plot_utils.plot_gene_composition(y.mean(axis=0), Model.data, gen.boxes[0], n_replicates, Model.sd)
    else:
        verdict = ' do not correlate significantly'
    print('Results of two independent runs' + verdict)

def parse_gtf(path, type='transcript'):
    gtf = pd.read_csv(path, sep='\t', names=['chrom', '_', 'type', 'start', 'end', '-', 'strand', '--', '---'])
    table = gtf.loc[:,['chrom', 'type', 'start', 'end', 'strand']]
    table = table[table['type']==type].loc[:,['chrom', 'start', 'end', 'strand']]
    if type == 'transcript':
        print('WARNING: using transcripts as genes. Keep in mind - starts and stops of transcripts are not starts and stops of genes')
    return table

def parse_tsv(path, score_col='score'):
    bed = pd.read_csv(path, sep='\t')
    table = bed.loc[:,['chrom', 'start', 'end', 'strand', score_col]]
    table = table.rename(columns={score_col:'score'})
    return table

def parse_bed(path):
    bed = pd.read_csv(path, sep='\t', names=['chrom', 'start', 'end', '-' 'strand', 'score'])
    table = bed.loc[:,['chrom', 'start', 'end', 'strand', 'score']]
    return table

def load_gene_table(data, path, type='transcript'):
    '''Loads table withe gene information'''
    if path.endswith('.gtf') or path.endswith('.gtf.gz'):
        gene_table = parse_gtf(path, type)
    elif path.endswith('.bed'):
        gene_table = parse_bed(path)
    elif path.endswith('.tsv') or path.endswith('.csv'):
        gene_table = parse_tsv(path)
    else:
        raise ValueError('gene annotation file extesnsion not recognized')
    mask = np.full(len(gene_table), False, dtype=bool)
    for chrom in data.chromnames:
        mask |= gene_table.chrom==chrom
    gene_table= gene_table[mask]
    gene_table = gene_table.sort_values(by=['chrom', 'start'])
    gene_table = gene_table.drop_duplicates()
    gene_table.data_name = 'Genes'
    return gene_table

def make_upstream_regions_table(table, upstream, downstream):
    '''Makes upstream regions for given regions'''
    g = table.copy()
    g.loc[g.strand=='+', 'end'] = g.start[g.strand=='+'] + downstream
    g.loc[g.strand=='+', 'start'] -= upstream
    g.loc[g.strand=='-', 'start'] = g.end[g.strand=='-'] - downstream
    g.loc[g.strand=='-', 'end'] += upstream
    return g

def make_downstream_regions_table(table, upstream, downstream):
    '''Makes downstream regions for given regions'''
    g = table.copy()
    g.loc[g.strand=='-', 'end'] = g.start[g.strand=='-'] + upstream
    g.loc[g.strand=='-', 'start'] -= downstream
    g.loc[g.strand=='+', 'start'] = g.end[g.strand=='+'] - upstream
    g.loc[g.strand=='+', 'end'] += downstream
    return g

def flip_annotation(table):
    '''Simple way of making annotation random for control and saving length and\
 distance distributions'''
    table = table.copy()
    for chrom in table['chrom'].unique():
        chrom_mask = table['chrom']==chrom
        end = table[chrom_mask]['end'].max()
        table.loc[chrom_mask, 'start'] = end - table.loc[chrom_mask, 'start']
        table.loc[chrom_mask, 'end'] = end - table.loc[chrom_mask, 'end']
    table = table.rename(columns={'start':'end', 'end':'start'})
    table = table.sort_values(by=['chrom', 'start'])
    return table

def make_subtable(table, regions):
    '''Select only regions belonging to specified regions'''
    mask = np.full(len(table), False)
    for region in regions:
        chrom, start, end = region
        mask |= (table['chrom']==chrom) & (table['end'] > start) & (table['start'] < end)
    return table[mask].copy()
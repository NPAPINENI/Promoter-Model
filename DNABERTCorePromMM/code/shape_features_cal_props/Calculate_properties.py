import pandas as pd
import sys

df_li=[]
region=sys.argv[1] #enhancer/non-enhancer
infile=sys.argv[2] #input file
outfile=sys.argv[3] #output file

en_df=pd.read_csv(infile, sep="\t")
en_df['Sequence']=en_df['Sequence'].str.upper()
# en_df['pos']=en_df['chr_38'] + ":" + en_df['start_38'].astype(str) + "-" + en_df['end_38'].astype(str)
#print(chr_name,len(en_df))
def label_to_region(label):
    if label == 1:
        return 'nonPromoter'
    else:
        return 'nonPromoter'

# print(len(sequence))

def calculate_A(sequence):
    sequence = row['Sequence'][10:21]
    label = row['Label']
    kmers=row['Kmers']
    seq_len=len(sequence)
    A_count = C_count = G_count = T_count = GC_count = CG_count = ACG_count = AGC_count = CAG_count = CCG_count = CGA_count = CGC_count = CGG_count = CGT_count = CTG_count = GAC_count = GCA_count = GCC_count = GCG_count = GCT_count = GGC_count = GTC_count = TCG_count = TGC_count = 0

    for nucleotide in sequence:
        if nucleotide == 'A':
            A_count += 1
        elif nucleotide == 'C':
            C_count += 1
        elif nucleotide == 'G':
            G_count += 1
        elif nucleotide == 'T':
            T_count += 1

    GC_count = sequence.count("GC")
    CG_count = sequence.count("CG")
    ACG_count = sequence.count("ACG")
    AGC_count = sequence.count("AGC")
    CAG_count = sequence.count("CAG")
    CCG_count = sequence.count("CCG")
    CGA_count = sequence.count("CGA")
    CGC_count = sequence.count("CGC")
    CGG_count = sequence.count("CGG")
    CGT_count = sequence.count("CGT")
    CTG_count = sequence.count("CTG")
    GAC_count = sequence.count("GAC")
    GCA_count = sequence.count("GCA")
    GCC_count = sequence.count("GCC")
    GCG_count = sequence.count("GCG")
    GCT_count = sequence.count("GCT")
    GGC_count = sequence.count("GGC")
    GTC_count = sequence.count("GTC")
    TCG_count = sequence.count("TCG")
    TGC_count = sequence.count("TGC")


    A_fract=A_count/seq_len
    C_fract=C_count/seq_len
    G_fract=G_count/seq_len
    T_fract=T_count/seq_len
    purpyr_fract=(A_count+G_count-C_count-T_count)/seq_len
    amke_fract=(A_count+C_count-G_count-T_count)/seq_len
    west_fract=(A_count+T_count-C_count-G_count)/seq_len
    cpg1 = (2*CG_count + 2*GC_count)/(seq_len-1)
    cpg2 = (ACG_count + AGC_count + CAG_count + CCG_count + CGA_count + CGC_count + 2* CGG_count + CGT_count + CTG_count + GAC_count + GCA_count + 2* GCC_count + GCG_count + GCT_count + 2* GGC_count + GTC_count + TCG_count + TGC_count)/(seq_len-2)
    cpg3 = (4* CAG_count + CCG_count + CGG_count + 4* CTG_count + 4* GAC_count + GCC_count + GGC_count + 4* GTC_count)/(seq_len-2)

    d={'Sequence':[sequence],'kmers': [kmers],'Label': [label], 'region':[region], 'A_Fraction':[A_fract], 'C_Fraction':[C_fract],'G_Fraction':[G_fract],'T_Fraction':[T_fract], 'PurPyr_Fraction':[purpyr_fract], 'AmKe_Fraction':[amke_fract], 'WeSt_Fraction':[west_fract], 'CpG1':[cpg1], 'CpG2':[cpg2], 'CpG3':[cpg3]}
    basedf=pd.DataFrame(data=d)   
    return basedf

base_df = pd.DataFrame()
results = []

for index, row in en_df.iterrows():
    result = calculate_A(row)
    results.append(result)

# Concatenate all results into a single DataFrame
if results:
    base_df = pd.concat(results, ignore_index=True)

# Apply the function to create a new column
base_df['region'] = en_df['Label'].apply(label_to_region)
# Merge selected original columns from en_df into base_df
# original_columns = ['Sequence', 'kmers', 'Label', 'SeqName', 'MGW_mean_10_20', 'HelT_mean_10_20', 'ProT_mean_10_20', 'Roll_mean_10_20', 'EP_mean_10_20']
original_columns = [col for col in en_df.columns if col in en_df.columns]

base_df = pd.concat([base_df.reset_index(drop=True), en_df[original_columns].reset_index(drop=True)], axis=1)
base_df.to_csv(outfile+region+"train_kmer_sequence.tsv",sep="\t", index=False)
base_df.drop(columns=['Sequence'], inplace=True)
base_df.to_csv(outfile+region+"train_kmer_properties.tsv", sep="\t", index=False)


# base_df.to_csv(outfile+region+"_kmer_sequence_train.tsv",sep="\t", index=False)
# base_df.drop(columns=['Sequence'], inplace=True)
# # base_df.to_csv(outfile+region+"_kmer_properties_train.tsv", sep="\t", index=False)

# base_df.to_csv(outfile+region+"train_kmer_sequence.tsv",sep="\t", index=False)
# base_df.drop(columns=['Sequence'], inplace=True)
# base_df.to_csv(outfile+region+"train_kmer_properties.tsv", sep="\t", index=False)

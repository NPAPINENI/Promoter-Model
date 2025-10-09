
# Pattern(TATA Box) based Promoter Model
The pattern-based datasets are curated with sequences that are identified with specific motifs (patterns) in a focused region. 
Here, promoter sequences are identified from transcription start sites (TSS) annotated in the GRCh38 genome by GENCODE. 
Promoter regions are defined as ±45 base pairs around each TSS, resulting in sequences of 90 base pairs. 
Focusing on the -35 to -25 bp region relative to the TSS, where the presence of the TATA-box motif exists (TATAWAW, where W = A or T) is analyzed.
Promoters containing this motif are labeled as "TATA" promoters, while those without are classified as "non-TATA" promoters.
# PWM (Position Weight Matrix) based Promoter Model
Position Weigth Matrix based Promoter models are curated using a scoring matrix representing the likelihood of each nucleotide (A, T, C, G) at every position within a motif. 
These matrices are typically derived from the observed frequencies of each nucleotide at each position, often obtained from resources like JASPAR through multiple sequence alignments and can be expressed as motif logos or probability matrices The frequencies are transformed into log-likelihood scores. 

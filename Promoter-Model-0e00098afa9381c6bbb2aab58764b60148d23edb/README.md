
# Pattern(TATA Box) based Promoter Model
This dataset contains human promoter sequences curated by searching for predefined sequence motifs in a focused region around annotated transcription start sites (TSS).
## Source
Genome: GRCh38 (human)

Annotations: GENCODE TSS coordinates
## Region
For each TSS, we extract ±45 bp around the TSS (positions −45…+45), yielding a 90 bp promoter sequence.

Motif focus: TATA-box

We examined the −35 to −25 bp window (relative to the TSS).

Motif tested: TATAWAW (where W ∈ {A, T}).
## Labels

TATA: promoter sequences that contain the TATAWAW motif within −35…−25.

non-TATA: promoter sequences that do not contain the motif within that window.



# PWM (Position Weight Matrix) based Promoter Model
PWM based models are curated using a scoring matrix representing the likelihood of each nucleotide (A, T, C, G) at every position within a motif. 

## Genome, region, and scanning
Genome: GRCh38 (human)

TSS source: GENCODE

Promoter sequence: ±45 bp around each TSS (total 90 bp, positions −45…+45).

Motif window: We evaluate the canonical TATA-box (via a TBP/TATA PWM) in the −35 to −25 bp region relative to TSS.

Strand: Both forward and reverse-complement strands are considered
## Labeling
TATA: promoter sequences with PWM score ≥ threshold within −35…−25.
non-TATA: no position in −35…−25 passes the threshold.
## Thresholding 
Gold standard: Eukaryotic Promoter Database (EPD).
Window: −35 to −25 bp relative to the TSS (canonical TATA-box location).
The 0.75 cutoff was selected by benchmarking against EPD to balance sensitivity and specificity for TATA detection in the canonical window.

# Funetuned CorePromMM model
The Funetuned CorePromMM model leverages the pretrained DNABERT architecture.
Please follow the structure below when setting up your repository:

├── DNABERTCorePromMM
│   ├── code  (for flexible usage, add these required py files and bash script in examples/ folder in your DNABERT repo)
│   │   ├── DNABERT-CoreProm_finetune.sh
│   │   ├── DNABERT-CoreProm_pretrained.py
│   │   ├── pretrain.sh
│   │   ├── run_finetune_DNABERT-CoreProm.py
│   │   ├── shape_features_cal_props   # feature calculation (Nucleotide Properties and Shape Features)
│   │   │   ├── Calculate_properties.py    # Run Calculate_properties.py and 500bp.R for Nucleotide Properties and Shape Features
│   │   │   ├── 500bp.R                  
│   │── src   # modified Functions from DNABERT (usage: download the DNABERT GitHub repo and replace this src/ folder with that)
│   │   ├── dnabert_transformers
│   │   └── transformers.egg-info
│   ├── data     # Curated data sets are in this data/ folder
│   │   ├── data_making
│   │   └── Promoter_data
│   │       ├── patternbased_TATA
│   │       ├── PB_NONTATA
│   │       ├── Promoter_vs_non_prom
│   │       ├── PWM_NONTATA
│   │       └── PWM_TATA
│   └── README.md
└── README.md


# ⚙️ Usage

Ensure that the latest src/ folder is present in the repository, containing the up-to-date DNABERT source code.
Use the DNABERT-CoreProm_finetune.sh script to run fine-tuning. This script automatically calls run_finetunre_DNABERT-CoreProm.py, which implements the FCPMM fine-tuning logic.

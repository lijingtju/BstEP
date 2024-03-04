


# BstEP
This repository contains machine learning models and prediction code for BstEP. Users can utilize BstEP to search for potential broad-spectrum antiviral drugs. We envision that BstEP will help accelerate the screening process for broad-spectrum drugs.
![flowchart BstEP.](https://github.com/lijingtju/BstEP/blob/main/BstEP_flowchart.png)

## Requirements
At the moment, a standard machine with CPUs will work.

## Installation
Currently, we suggest to run BstEP from source.

### Manual Start
```
git clone https://github.com/lijingtju/BstEP.git
cd /path/to/BstEP
conda env create -f BstEP.yml
conda activate BstEP
```
OR if you want to create the BstEP enviroment to a specific path, then
```
git clone https://github.com/lijingtju/BstEP.git
cd /path/to/BstEP
conda env create -f BstEP.yml --prefix /your/specif/path
conda env list
conda activate BstEP/in/your/env/list
```

### Commands to do prediction
python BstEP_predict.py --csvfile ./data/test.csv --outfile ./data/test_result.csv


### The detail of results.
The meaning of each column in results filesL:
```
1. Column "SMILES_stand": SMILES format of small molecule compounds after standardization.
2. Column "H1N1_pred" column indicates whether a small molecule compound possesses anti-H1N1 capabilities. "Active" denotes that BstEP recognizes the compound as having anti-H1N1 activity, while "in-active" indicates that BstEP determines the compound lacks such activity.
3. Column "H1N1_prob": Indicates the level of activity against the H1N1 virus. The higher the value of H1N1_prob, the greater the antiviral capacity of the small molecule compound against the H1N1 virus. In BstEP, a value of H1N1_prob greater than 0.5 is considered indicative of antiviral activity against the H1N1 virus.
4. Column "SARS_CPE_pred" column indicates whether a small molecule compound possesses anti-SARS-CoV-2 capabilities. "Active" denotes that BstEP recognizes the compound as having anti-SARS-CoV-2 activity, while "in-active" indicates that BstEP determines the compound lacks such activity.
5. Column "SARS_CPE_prob": Indicates the level of activity against the SARS-CoV-2 virus. The higher the value of SARS_CPE_prob, the greater the antiviral capacity of the small molecule compound against the SARS-CoV-2 virus. In BstEP, a value of SARS_CPE_prob greater than 0.5 is considered indicative of antiviral activity against the SARS-CoV-2 virus.
6. Column "EV71_pred" column indicates whether a small molecule compound possesses anti-EV-A71 capabilities. "Active" denotes that BstEP recognizes the compound as having anti-EV-A71 activity, while "in-active" indicates that BstEP determines the compound lacks such activity.
7. Column "EV71_prob": Indicates the level of activity against the EV-A71 virus. The higher the value of EV71_prob, the greater the antiviral capacity of the small molecule compound against the EV-A71 virus. In BstEP, a value of EV71_prob greater than 0.5 is considered indicative of antiviral activity against the EV-A71 virus.
```

### Note:
```
1. The *.csv files should contain the SMILES of small molecule compounds.
2. BstEP will automatically discard small molecule compounds that cannot be standardized.
3. When all three viruses recognize a small molecule compound as having antiviral activity, BstEP will consider this small molecule compound as a broad-spectrum antiviral drug candidate.
```

## Online BstEP webpage
You can visit our online page at [BstEP_web](http://www.BstEP.top/), which stores the predicted results of more than 300,000 small molecule compounds.

# Artificial Text Detection via Examining the Topology of Attention Maps 

This repository contains adaptation of code from "Artificial Text Detection via Examining the Topology of Attention Maps" presented at EMNLP 2021 conference for russian texts.

We briefly list the features below, and refer the reader to the paper for more details:
* Topological features (Betti numbers, the number of edges, the number of strong connected components, etc.);
* Features derived from barcodes (the sum of lengths of bars, the variance of lengths of bars, the time of birth/death of the longest bar, etc.);
* Features based on distance to patterns (attention to previous token, attention to punctuation marks, attention to CLS-token, etc.).

# Dependencies
The code base requires:
* python 3.8.3
* matplotlib 3.3.1
* networkx 2.5.1
* numpy 1.19.1
* pandas 1.1.1
* ripserplusplus 1.1.2
* scipy 1.5.2
* sklearn 0.23.2
* tqdm 4.46.0
* transformers 4.3.0


# Usage
* For calculating topological invariants by thresholds (Section 4.1), use `features_calculation_by_thresholds.ipynb`.
* For calculating barcodes (Section 4.2) and template (Section 4.3) features, use `features_calculation_barcodes_and_templates.ipynb`.
* For making predictions with the logistic regression upon calculated features, use `features_prediction_gpt_web.ipynb`.
* The head-wise probing analysis by [Jo and Myaeng (2020)](https://aclanthology.org/2020.acl-main.311/) is conducted using [an open-source implementation](https://github.com/heartcored98/transformer_anatomy). 



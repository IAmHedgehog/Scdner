# ScdNER: Span-Based Consistency-Aware Document-Level Named Entity Recognition
PyTorch code for ScdNER: "ScdNER: Span-Based Consistency-Aware Document-Level Named Entity Recognition". For a description of the model and experiments, see our paper: [https://par.nsf.gov/servlets/purl/10515191](https://par.nsf.gov/servlets/purl/10515191) (published at EMNLP 2023).

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)

## Examples
(1) Train SciERC on train dataset, evaluate on dev dataset:
```
python ./spert.py train --config configs/SciERC/train.conf
```

(2) Evaluate the SciERC model on test dataset:
```
python ./spert.py eval --config configs/SciERC/eval.conf
```

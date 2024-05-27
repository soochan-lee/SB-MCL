# Learning to Continually Learn with the Bayesian Principle

This repository contains the code for our ICML 2024 paper titled *Learning to Continually Learn with the Bayesian Principle*.

## Requirements

- Python 3.10
- Pip packages:
```bash
pip install -r requirements.txt
```

## Usage

The basic usage of the training script is as follows:
```bash
python train.py -c [config] -o [override options] -l [log directory]
```
In `cfg/`, we provide the configuration files for all the experiments in the paper.

After training, we evaluate the models using the following command:
```bash
python evaluate.py -l [log directory]
```
The SB-MCL (MAP) scores can be attained by turning on the `map` option.
```bash
python evaluate.py -l [SB-MCL log directory] -o "map=True"
```

## Datasets

All datasets except MS-Celeb-1M are downloaded automatically by the code.
Note that downloading the CASIA dataset may take days.

### MS-Celeb-1M

Use BitTorrent to download the dataset from [Academic Torrents](https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97).
```bash
transmission-cli https://academictorrents.com/download/9e67eb7cc23c9417f39778a8e06cca5e26196a97.torrent -w data
```

# FACT-UVA

## IMPORTANT !!!

Repo moved from https://github.com/fricer/FACT-UVA to https://github.com/YovchevIvan/FACT-UVA in order to be able to add more than 4 collaborators

## Links

Debiaswe: https://github.com/tolga-b/debiaswe
Lipstick: https://github.com/gonenhila/gender_bias_lipstick

### How to get the GoogleNews word2vec embeddings:
Download it directly from the official [website](https://code.google.com/archive/p/word2vec/) or clone [this github repo](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). Place the downloaded **.bin** file in the embeddings folder.

### To run debias on embeddings do:
```
python3 main.py ../embeddings/GoogleNews-vectors-negative300.bin ../data/definitional_pairs.json ../data/gender_specific_full.json ../data/equalize_pairs.json ../embeddings/GoogleNews-vectors-negative300-hard-debiased.bin
```



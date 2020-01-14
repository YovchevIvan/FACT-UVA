# FACT-UVA: Man is to Programmer as Woman is to Homemaker?

## IMPORTANT !!!

Repo moved from https://github.com/fricer/FACT-UVA to https://github.com/YovchevIvan/FACT-UVA in order to be able to add more than 4 collaborators

## Links

Debiaswe: https://github.com/tolga-b/debiaswe
Lipstick: https://github.com/gonenhila/gender_bias_lipstick

### How to get the GoogleNews word2vec embeddings:
Download it directly from the official [website](https://code.google.com/archive/p/word2vec/) or clone [this github repo](https://github.com/mmihaltz/word2vec-GoogleNews-vectors). Place the downloaded **.bin** file in the embeddings folder.

### How to get the Glove embeddings:
Go to the official [website](https://nlp.stanford.edu/projects/glove/). Download **glove.840B.300d.zip**. Place the downloaded **.txt** file in the embeddings folder.

### To run debias on word2vec embeddings do:
```
python3 main.py
```
where __--em_limit__ argument can be used to limit the number of words being loaded

### To run debias on glove embeddings do:
```
./gloveToW2V.sh ../embeddings/glove.840B.300d.txt ../embeddings/glove.formatted.txt
python3 main.py --bin=False --i_em=../embeddings/glove.formatted.txt
```
where __--em_limit__ argument can be used to limit the number of words being loaded


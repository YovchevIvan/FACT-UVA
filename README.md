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

By default the script uses the file __GoogleNews-vectors-negative300.bin__ to read from so it does not need to be specified. When using a diffrent embeddings file please use the __--i_em__ argument to specify it. Additionally, the __--em_limit__ argument can be used to limit the number of words being loaded, by default this is set to 50K words as specified in the original paper. If needed the resulting vector files can be saved in **.txt** format using the __o_ext__ argument. However, for perfromance reasons the default is set to **.bin**

```
python3 main.py --debias_o_em=../embeddings/debiased_word2vec.bin --bias_o_em=../embeddings/bias_word2vec.bin
```

### To run debias on glove embeddings do:

If running for the first time do:
```
./gloveToW2V.sh ../embeddings/glove.840B.300d.txt ../embeddings/glove.formatted.txt
```

then to run do:

```
python3 main.py --i_em=../embeddings/glove.formatted.txt --debias_o_em=../embeddings/debiased_glove.bin --bias_o_em=../embeddings/bias_glove.bin
```

### To run analogy generator

Run the script in `FACT-UVA/code/analogies.py` as follows

```
python3 analogies.py --i_em=<path to embeggindg gile> --complete x-y-z
```

Where `x`, `y`, and `z` are words. The script will then find a word `w` such that `x:y=z:w`. Alternatively the script can be run as follows:

```
python3 analogies.py --i_em=<path to embeggindg gile> --pair_seed x-y
```

Where `x` and `y` are again words. The script will then find a pair `(z,w)` such that `x:y=z:w`.

Additional arguments can be given such as `--em_limit` followed by the maximum number of embeddings to be loaded, and `--bin`, followed by a boolean value, to specify weather or not the loaded embeddings file is binary or not (txt).

### To run bench-marking tests:

First install the benchmarking tools provided by [this github repository](https://github.com/kudkudak/word-embeddings-benchmarks). To do so move to the directory `FACT-UVA/code/benchmarks` and run 

```
python3 setup.py install
```

Further instruction regarding the package installation can be found in `FACT-UVA/code/benchmarks/README.rst`. Once this is done, the testing script can be run with the following command from the directory `FACT-UVA/code/benchmarks/scripts`:

```
python3 evaluate_on_all.py -f <embeddings binary file>
```

The latter is the name of the output dumped by the script `FACT-UVA/code/main.py` after debiasing given embeddings.

To run all benchmark tests run
```
./run_test.sh
```

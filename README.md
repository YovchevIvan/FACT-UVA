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
./gloveToW2V.sh ../../embeddings/glove.840B.300d.txt ../../embeddings/glove.formatted.txt
```

then to run do:

```
python3 main.py --i_em=../embeddings/glove.formatted.txt --debias_o_em=../embeddings/debiased_glove.bin --bias_o_em=../embeddings/bias_glove.bin
```

#### To run professions projecting on a given axis

You can run the __main.py__ script as described above with additional parameters. __--load_profs__ must be set to True to call the projecting method. A list of professions can be specified through __--profs__ param where the value should be the path to a JSON file. By default the `data/professions.json` file is used so does not need to be specified. The __--axis_profs__ param is used to specify the two words defining the axis to project on. Default is softball-football. The format to specify the axis in is: word1-word2. Finally, the __--n_profs__ param can be used to specify how many of the profession extremes to print. The default is 5 as seen in Figure 3 in the [original paper](https://arxiv.org/abs/1607.06520)

### To run analogy generator

Run the script in `FACT-UVA/code/analogies.py` as follows

```
python3 analogies.py --i_em=<path to embeggindg file> --complete x-y-z
```

Where `x`, `y`, and `z` are words. The script will then find a word `w` such that `x:y=z:w`. Alternatively the script can be run as follows:

```
python3 analogies.py --i_em=<path to embeggindg file> --pair_seed x-y
```

Where `x` and `y` are again words. The script will then find a pair `(z,w)` such that `x:y=z:w`. If a number `--n` is specified, then that number of analogies are generated (by sampling random `z` and solving `x:y=z:w` for `w`), otherwise if a file `--z_file` is given, then is is assumed to be a list of words to use as third element to complete an analogy from the pair seed. The parameter `--pairs_fname` followed by a file name, will be used to determine the file in which to dump the output. Finally, if more than one solution is given (if `--n` is specified), the pairs will be sorted by distance, and the top 10 closest one displayed as well.

Finally, a json file with a list of pairs can be given, to each be used as a generative pair for analogies. If so, the number of analogies must be also specified. To do so, run:

```
python3 analogies.py --i_em=<path to embeggindg file> --file_seed <path to json file> --n <number of analogies to generate>
```

The analogies are printed only, to store them you can run 

```
python3 analogies.py --i_em=<path to embeggindg file> --file_seed <path to json file> --n <number of analogies to generate> 1> <output file name>
```

Additional arguments can be given such as `--em_limit` followed by the maximum number of embeddings to be loaded .

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

### To run classification
This part is based on the paper 'Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them'
paper: https://arxiv.org/abs/1903.03862
github: https://github.com/gonenhila/gender_bias_lipstick

Our script extends the classification test to run for severall classifiers, with increasing amounts of training data

Parameters: 

--embeddings_original: Path to the file containing the original embeddings (default: "../../embeddings/bias_word2vec.bin")

--embeddings_debiased: Path to the file containing the debiased embeddings (default: "../../embeddings/debiased_word2vec.bin")

--fname: the name of tag added to the plot output

Run (from code folder):
```
python3 classify_debiased.py --embeddings_original=original embeddings file path --embeddings_debiased=debiased embeddings file path
```

 


#!/bin/bash

## get full path to working dir
full_path=`pwd`

# remove suffix
suffix="code/benchmark/scripts"
full_path=${full_path%"$suffix"}

# get path to embeddings
full_path="${full_path}embeddings"

# declare an array variable
declare -a vecs=("bias_word2vec" "debiased_word2vec" "bias_glove" "debiased_glove")

# variable to store string
result=""

# new line character
NEWLINE=$'\n'

## now loop through the above array
iter=0

# for each type of vector
for i in "${vecs[@]}"
do
    ## run to get results
    python3 evaluate_on_all.py -f "${full_path}/$i.bin" -o "$i.csv"

    # if its the first iteration
    if [[ "$iter" -eq 0 ]];
    then
        ## get firt line with titles
        line=$(sed -n '1p' "$i.csv")

        ## append it to results
        result="${result}Name${line}${NEWLINE}"
    fi

    ## get second line
    line=$(sed -n '2p' "$i.csv")

    ## add embedding name to front
    line="${line:1:${#line}}"

    ## append to result
    result="${result}${i}${line}${NEWLINE}"

    # delete file
    rm "$i.csv"

    ## update iter count
    iter=$(expr $iter + 1)
done

# store resulting string to file
echo "$result" >> result.csv
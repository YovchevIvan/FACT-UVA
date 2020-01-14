#! /bin/bash

echo extracting number of vectors
nlines=$(cat $1 | wc -l)
echo there are $nlines lines

echo extracting vector dimension
dim=$(($(cat $1 | head -n 1 | wc -w)-1))
echo vectors have size $dim

echo creating word2vec format file
echo $nlines $dim > $2
cat $1 >> $2
echo done

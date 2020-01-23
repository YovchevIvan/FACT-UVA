#! /bin/bash

# get file content and count number of lines (thus number of embeddings)
echo extracting number of vectors
nlines=$(cat $1 | wc -l)
echo there are $nlines lines

# count number of words in the first line and subtract one to get embeddings dimentsion
echo extracting vector dimension
dim=$(($(cat $1 | head -n 1 | wc -w)-1))

# check if the file is already in word2vec format
if [ $dim -eq 1 ] #if there are only 2 (-1) words in the first line
then
	
	echo it seems that the file is already in word2vec format, checking...

  # get vector dimension counting the words in the second line insted
	echo recomputing dim from second line
	dim=$(($(cat $1 | head -n 2 | tail -n 1 | wc -w)-1))
	echo vectors have size $dim

  # from the first line get the specified number of lines and dimensions
	echo extracting reference sizes from format
	supposednlines=$(cat $1 | head -n 1 | cut -d ' ' -f1) # get specified number of lines
	supposeddim=$(cat $1 | head -n 1 | cut -d ' ' -f2) # get specified dimensions
	echo format specifies that there should be $supposednlines vectors of vectors of size $supposeddim

  # if actual number of lines or the actual embeddings direction do not coincide with the two numbers in the first line
	if [ $(( $nlines - 1 ))!=$supposednlines ] || [ $dim!=$supposeddim ]; then
    # then it is neither glove nor word2vec format
		echo number of lines or vector dim not coinciding
		exit 1
	fi

  # otherwise it is already in word2vec format
	echo file in word2vec format already
	exit 0
fi

echo vectors have size $dim

# convert to word to2vec format
echo creating word2vec format file
echo $nlines $dim > $2 # write number of embeedinds and their dim in the first line
cat $1 >> $2 # write embeddings after that
echo done

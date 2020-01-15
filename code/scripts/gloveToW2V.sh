#! /bin/bash

echo extracting number of vectors
nlines=$(cat $1 | wc -l)
echo there are $nlines lines

echo extracting vector dimension
dim=$(($(cat $1 | head -n 1 | wc -w)-1))

if [ $dim -eq 1 ]
then
	
	echo it seems that the file is already in word2vec format, checking...

	echo recomputing dim from second line
	dim=$(($(cat $1 | head -n 2 | tail -n 1 | wc -w)-1))
	echo vectors have size $dim

	echo extracting reference sizes from format
	supposednlines=$(cat $1 | head -n 1 | cut -d ' ' -f1)
	supposeddim=$(cat $1 | head -n 1 | cut -d ' ' -f2)
	echo format specifies that there should be $supposednlines vectors of vectors of size $supposeddim

	if [ $(( $nlines - 1 )) != $supposednlines ] || [ $dim != $supposeddim ]; then
		echo number of lines or vector dim not coinciding
		exit 1
	fi

	echo file in word2vec format already
	exit 0
fi

echo vectors have size $dim

echo creating word2vec format file
echo $nlines $dim > $2
cat $1 >> $2
echo done

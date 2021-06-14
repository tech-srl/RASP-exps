#!/bin/bash
shopt -s expand_aliases
alias python3='/usr/local/opt/python@3.8/bin/python3'

# exactly zero good reasons for using this script instead of calling python directly
# other than i got used to this and here we are

## example run: lang=reverse seq=abc contains=256_1_2 withnums=2 ./generic_draw.sh


[ -z "$contains" ] && contains="NOT-PASSED"
[ -z "$withnums" ] && withnums="-1"
# [ -z "$height" ] && height=5
# [ -z "$width" ] && width=5

if [ $contains == NOT-PASSED ]; then
	contains_seq=""
else
	contains_seq="--must-contain=$contains"
fi


python3 LoadAndDrawAttention.py --lang=$lang --sequence=$seq $contains_seq \
	 --with-numbers=$withnums 
	 # --height=$height --width=$width

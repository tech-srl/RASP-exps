#!/bin/bash

# honestly this is no different from the python file at this point. only really keeping it around for loops i think?

## example run: lang=hist2_26_bos_with_attns heads=2 layers=2 ./generic_run.sh

[ -z "$lr" ] && lr=0.0003
[ -z "$gamma" ] && gamma=0.98
[ -z "$position" ] && position=sin # either sin or embed.
[ -z "$attn_c" ] && attn_c=100
[ -z "$dropout" ] && dropout=0.0
[ -z "$reset_patience" ] && reset_patience=10

[ -z "$small" ] && small=false
[ -z "$overfit" ] && overfit=false
[ -z "$tinyset" ] && tinyset=0
[ -z "$traincrop" ] && traincrop=0

[ -z "$heads" ] && heads=1
[ -z "$dim" ] && dim=128
[ -z "$layers" ] && layers=1

[ -z "$epochs" ] && epochs=50
[ -z "$batch_size" ] && batch_size=50
[ -z "$subfolder" ] && subfolder=""

ff=$((2*$dim))

if [ $small == true ]; then
	heads=4
	dim=16
	ff=20
	layers=2
	tinyset=100
	echo making small model
fi

optim="adam"

if [ $overfit == false ]; then
	tinyflag="--tiny=$tinyset"
else
	tinyflag="--overfit"
fi




echo training with $LAYERS layers
echo running with tinyflag: $tinyset

python3 MakeTransformer.py --lang=$lang --nheads=$heads --d-model=$dim \
--nlayers=$layers --ff=$ff \
--initial-lr=$lr --reset-patience=$reset_patience --n-epochs=$epochs --batch-size=$batch_size \
--subfolder=$subfolder --optim=$optim --gamma=$gamma \
--positional-encoding-type=$position $tinyflag --crop-train=$traincrop \
--attn-loss-c=$attn_c --dropout=$dropout

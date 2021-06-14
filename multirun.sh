#!/bin/bash

# example: lang=reverse_100 heads=1 layers=2 ./multirun.sh

epochs=100
drop=0
d=256
dX2=$((2*$d))
headsX2=$((2*$heads))
headsM1=$(($heads-1))
layersM1=$(($layers-1))
sf="example"

for lr in 0.0003 0.0001
do
for g in 0.98 0.99
do
    lang=$lang heads=$heads layers=$layers dim=$d epochs=$epochs dropout=$drop gamma=$g lr=$lr subfolder=$sf ./generic_run.sh
    
    # test: drop one head
    if [ $headsM1 -gt 0 ] 
    then 
      lang=$lang heads=$headsM1 layers=$layers dim=$d epochs=$epochs dropout=$drop gamma=$g lr=$lr subfolder=$sf ./generic_run.sh
    fi

    # test: drop one layer
    if [ $layersM1 -gt 0 ] 
    then 
      lang=$lang heads=$heads layers=$layersM1 dim=$d epochs=$epochs dropout=$drop gamma=$g lr=$lr subfolder=$sf ./generic_run.sh
      # verify not actually seeing effects of dropping one head, i.e. it is the layer itself that is necessary, 
      # by giving more heads (and dim to hold them):
      lang=$lang heads=$headsX2 layers=$layersM1 dim=$dX2 epochs=$epochs dropout=$drop gamma=$g lr=$lr subfolder=$sf ./generic_run.sh
    fi
done    
done






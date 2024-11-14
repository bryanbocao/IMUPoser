#!/bin/bash

combos=0

for combo in $combos
do 
  echo Running combo $combo
  python 1_train_global_model.py --combo_id $combo --experiment 'IMUPoserGlobalModel'
done

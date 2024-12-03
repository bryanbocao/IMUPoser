#!/bin/bash

combos='global'

for combo in $combos
do 
  echo Running combo $combo
  CUDA_VISIBLE_DEVICES=1 python 1_train_global_model.py --combo_id $combo --experiment 'IMUPoserGlobalModel'
done

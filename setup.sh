#!/bin/bash

git submodule init
git submodule update

#python3 install -r requirements.txt

mkdir data

kaggle competitions download -c quora-question-pairs
mv quora-qustion-pairs.zip data
unzip ./data/quora-question-pairs

gdown https://drive.google.com/uc\?id\=1uh3VaWkUCNj9S3-1Uv63CClrEo6WP7cu ./models/clf_head_weight
gdown https://drive.google.com/uc\?id\=1wHXh7Gn1GbPpsva_0tHm2ybpoJ7AHUfJ ./models/checkpoint_iter_24983_2019-11-04 03:37:20.323658

#!/usr/bin/env bash

wget https://www.ariel-datachallenge.space/static/data/ml_data_challenge_database.zip
unzip ml_data_challenge_database.zip

mkdir -p data/params_train
mkdir data/noisy_test
mkdir data/noisy_train

tar -xvf ml_data_challenge_database/params_train.tar -C data/params_train/
tar -xvf ml_data_challenge_database/noisy_test.tar -C data/noisy_test/
tar -xvf ml_data_challenge_database/noisy_train.tar -C data/noisy_train/

rm ml_data_challenge_database.zip
rm ml_data_challenge_database/params_train.tar
rm ml_data_challenge_database/noisy_test.tar
rm ml_data_challenge_database/noisy_train.tar 
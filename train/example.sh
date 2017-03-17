#!/bin/bash

port=2936
cur_dir=$(pwd)

train_file=$cur_dir/test/train
feature_file=$cur_dir/test/features 
output_dir=$cur_dir/test/ 

sh train.sh $train_file $feature_file $output_dir $port

# after starting the service, use the below code to do test, and kill the server
# sh predict.sh $port
# sh kill.sh $port

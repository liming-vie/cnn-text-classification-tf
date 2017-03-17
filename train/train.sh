#!/bin/bash
if [ $# != 4 ] ; then
	echo -e "\033[31m sh train.sh feature_file train_file output_dir service_port\033[0m"
	exit 1
fi

# train_file:
#	each line with query title label
#	use utf-8 encoding
#	use \t to split each field
#	labe: [0/1]
train_file=$1
# feature_file:
#	each line with title features
#	use \t to split each field
#	use spacing to split numerical features
feature_file=$2
# for save output_files
output_dir=$3
# for start thrift service
service_port=$4

la_port=1028 # la segmentation service 

mkdir $output_dir

echo -e "\033[31m generating feature vector in $output_dir/data_vec_[pos/neg]\033[0m"
vec_dim=$(python feature.py $train_file $feature_file $output_dir/data_vec)

echo -e "\033[31m generating input data in $output_dir/data_[pos/neg]\033[0m"
python query.py $train_file $output_dir/data $la_port

script_dir=$(pwd)
text_cnn_dir=$(pwd)/../textcnn/
cd $text_cnn_dir
echo -e "\033[31m now in directory $text_cnn_dir\033[0m"

echo -e "\033[31m generating vocabulary in $output_dir/vocab_str\033[0m"
python vocab.py --positive_data_file=$output_dir/data_pos --negative_data_file=$output_dir/data_neg > $output_dir/vocab_str
python $script_dir/clean_vocab_str.py $output_dir/vocab_str $output_dir/vocab_str_clean # remove useless output info
mv $output_dir/vocab_str_clean $output_dir/vocab_str

echo -e "\033[31m getting pre-trained word vector in $output_dir/vocab_vec\033[0m"
python $script_dir/generate_vocab_vec.py $output_dir/vocab_str $output_dir/vocab_vec

echo -e "\033[31m start train textcnn, enter Ctrl-C when about converge\033[0m"
mkdir $output_dir/train.save
python train.py --positive_data_file=$output_dir/data_pos --negative_data_file=$output_dir/data_neg --positive_vec_file=$output_dir/data_vec_pos --negative_vec_file=$output_dir/data_vec_neg --vocab_embedding_file=$output_dir/vocab_vec --output_dir=$output_dir/train.save
python $script_dir/check_model.py $output_dir/train.save/checkpoints/
echo -e "\033[31m training result saved in $output_dir/train.save\033[0m"

echo -e "\033[31m start thrift service in port $service_port, can use sh kill.sh $service_port to kill it\033[0m"
cd server
nohup python disab_server.py --port=$service_port --checkpoint_dir=$output_dir/train.save/checkpoints/ --title_feature_file=$feature_file --la_port=$la_port &

echo -e "\033[31m back in directory $script_dir\033[0m"
cd $script_dir

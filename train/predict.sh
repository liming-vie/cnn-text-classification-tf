#!/bin/bash
if [ $# != 1 ] ; then
	echo 'sh predict.sh port_number'
	exit 1
fi
	
cur_dir=$(pwd)
server_dir=$(pwd)/../textcnn/server/
port=$1

echo 'please input your query line by line'
cd $server_dir
python disab_client.py $port

cd $cur_dir

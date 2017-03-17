#!/bin/bash
if [ $# != 1 ] ; then 
	echo -e "\033[31m use sh kill.sh port_number\033[0m"
	exit 1
fi

echo -e "\033[31m kill service in port $1\033[0m"
kill -9 $(ps aux | grep "python disab_server.py \-\-port=$1" | awk '{print $2}')

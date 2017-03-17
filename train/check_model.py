#! /usr/bin/env python
import sys
import os
checkpoint_dir=sys.argv[1]

with open(checkpoint_dir+'checkpoint') as fin:
	latest_model=fin.readlines()[-1].strip().split(' ')[-1][1:-1]

suffixs=['meta', 'index', 'data-00000-of-00001']
for suffix in suffixs:
	if not os.path.isfile(latest_model+'.'+suffix):
		os.system("sed -i '$d' "+checkpoint_dir+'checkpoint')
		break

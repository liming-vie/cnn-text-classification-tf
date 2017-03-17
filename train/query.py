#! /usr/bin/env python
import sys
sys.path.append('../la')
sys.path.append('../la/gen-py')
from la_client import LAClient

input_file=sys.argv[1]
output_file=sys.argv[2]
la_port=int(sys.argv[3])

client=LAClient(la_port)
pos_vec=[]
neg_vec=[]
with open(input_file) as fin, open(output_file+'_neg', 'w') as negout, open(output_file+'_pos', 'w') as posout:
	for line in fin:
		ps=line.strip().split('\t')
		vec=neg_vec if ps[2]=='0' else pos_vec
		vec.append(ps[0].lower().replace(ps[1].lower(), "MOVIE"))
	pos_vec=client.request_multiple(pos_vec)
	neg_vec=client.request_multiple(neg_vec)
	posout.write('\n'.join(pos_vec)+'\n')
	negout.write('\n'.join(neg_vec)+'\n')

#! /usr/bin/env python
vocab=open('../data/vocab_vec')
dic={}
for line in vocab:
	line=line.strip()
	ps=line.split(' ')
	dic[ps[0].lower()] = (' ').join(ps[1:])
vocab.close()

embedding_dim=len(dic[ps[0].lower()].split(' '))
#print embedding_dim

import sys
input_name=sys.argv[1]
output_name=sys.argv[2]

fin=open(input_name)
fout=open(output_name, 'w')
for line in fin:
	line=line.strip().lower()
	if line in dic:
		fout.write(dic[line]+'\n')
	else:
		fout.write(' '.join(['0' for i in range(embedding_dim)])+'\n')

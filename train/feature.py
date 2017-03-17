#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

fin_file=sys.argv[1]
feature_file=sys.argv[2]
fout_file=sys.argv[3]

# load term frequency
tf={}
with open('../data/term_frequency') as fin:
	for line in fin:
		ps=line.split('\t')
		tf[ps[0].lower().strip()]=int(ps[1])
# for normalization
mx3=0	# term frequncy
with open(fin_file) as fin:
	for line in fin:
		ps=line.split('\t')
		ps[1]=ps[1].lower().strip()
		t=tf[ps[1]] if ps[1] in tf else 0
		mx3=mx3 if t <= mx3 else t
# load title feature vector
feature={}
with open(feature_file) as fin:
	for line in fin:
		ps=line.strip().split('\t')
		feature[ps[0].strip().lower()]=ps[1].split(' ')
dim=len(open(feature_file).readline().strip().split('\t')[1].split(' '))

with open(fin_file) as fin, open(fout_file+'_neg','w') as negout, open(fout_file+'_pos', 'w') as posout:
	for line in fin:
		ps=line.strip().split('\t')
		try:
			fout=negout if ps[2]=='0' else posout
			title=ps[1].lower().strip()
			t = tf[title] if title in tf else 0
			# term frequency, query length, title length
			fout.write(str(float(t)/float(mx3))+'\t'+str(len(ps[0].strip().lower().decode('utf-8')))+'\t'+str(len(title.decode('utf-8'))))
			# user defined title features
			tmp = feature[title] if title in feature else ['0.0' for _ in range(dim)]
			fout.write('\t'+'\t'.join(tmp))
			fout.write('\n')
		except Exception, e:
			print >> sys.stderr, e

# print vector dimension
dim += 3
print dim

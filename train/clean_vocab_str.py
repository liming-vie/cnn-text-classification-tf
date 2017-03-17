#! /usr/bin/env python
import sys
input_file=sys.argv[1]
output_file=sys.argv[2]

lines=open(input_file).readlines()
with open(output_file, 'w') as fout:
	fout.write(''.join(lines[8:]))

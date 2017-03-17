#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('gen-py')

from disambiguation import Disambiguation
from disambiguation.ttypes import DisABResponse

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

class Client: 
	def __init__ (self, port):
		self.transport = TSocket.TSocket('localhost', port)
		self.transport = TTransport.TBufferedTransport(self.transport)
		protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

		self.client = Disambiguation.Client(protocol)

	def request(self,  query):
		self.transport.open()
		ret = self.client.run(query)
#		print len(ret), ret
		self.transport.close()
		return ret 

if __name__ == '__main__':
	port=int(sys.argv[1])
	client=Client(port)
	while True:
		query=raw_input()
		res=client.request(query)
		for t in res:
			print t.ibegin, t.iend, t.tag, t.score

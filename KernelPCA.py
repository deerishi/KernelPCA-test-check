import math as m
import sys
import numpy as n 
from numpy import *
import matplotlib.pyplot as p
#from tools.load import LoadMatrix

d=[[0 for x in xrange(101)] for x in xrange(2)] 
for x in range(-50,1):
  d[0][x+50]=x
for x in range(1,51):
	d[0][x+50]=x
		
i=0
while(i<51):
		#print i
	y2=2500-(i*i)
	yy2=m.sqrt(y2)
	d[1][i+50]=yy2
	i=i+1
i=50
while(i>0):
		#print 
	y2=2500-(i*i)
	yy2=m.sqrt(y2)
	d[1][50-i]=yy2
	i=i-1
		#d[1][100+x]=yy2
	
			
	
	#print d[0][:],'\n',d[1][:]
	#p.plot(d[0][:],d[1][:],'x')
	
d2=[[0 for x in xrange(201)] for x in xrange(2)] 
for x in range(-100,1):
	d2[0][x+100]=x
for x in range(1,101):
	d2[0][x+100]=x
		
i=0
while(i<101):
		#print i
	y2=10000-(i*i)
	yy2=m.sqrt(y2)
	d2[1][i+100]=yy2
	i=i+1
i=100
while(i>0):
		#print 
	y2=10000-(i*i)
	yy2=m.sqrt(y2)
	d2[1][100-i]=yy2
	i=i-1
		#d[1][100+x]=yy2
	#print d2[0][:],'\n',d2[1][:]
p.plot(d[0][:],d[1][:],'x',d2[0][:],d2[1][:],'o')
data=hstack((d,d2))
#data=d
print 'size =',n.shape(data)
p.show()

parameter_list = [[data,0.01,1.0], [data,0.05,2.0]]

def preprocessor_kernelpca_modular (data, threshold, width):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelPCA
	from shogun.Kernel import GaussianKernel
	#from shogun.Converter import StochasticProximityEmbedding, SPE_GLOBAL
	features = RealFeatures(data)
	
	kernel=GaussianKernel(features,features,width)
		
	preprocessor=KernelPCA(kernel)
	preprocessor.init(features)
	preprocessor.apply_to_feature_matrix(features)
	X = preprocessor.get_transformation_matrix()
	#l1=len(X)
	X2=preprocessor.apply_to_feature_matrix(features)
	print 'the rows=%d, ',n.shape(X2)
	p.plot(X2[0][:],'x')
	p.show()
	print 'type of features=%',(type(X))
	print 'X=\n',X2
	return features



if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca_modular(*parameter_list[0])

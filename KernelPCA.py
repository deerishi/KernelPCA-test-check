
import math as m
import sys
import numpy as n
from numpy import *
import matplotlib.pyplot as p
d=[[0 for x in xrange(50)] for x in xrange(2)] 
for x in range(50):
	d[0][x]=x
		
i=1
while(i<51):
	yy2=m.log(i)
	d[1][i-1]=yy2
	i=i+1
d2=[[0 for x in xrange(50)] for x in xrange(2)] 
for x in range(50):
	d2[0][x]=x		
i=1
while(i<51):
	yy2=m.log(i)-1
	d2[1][i-1]=yy2
	i=i+1
p.plot(d[0][:],d[1][:],'x',d2[0][:],d2[1][:],'o')
p.show()
data=hstack((d,d2))
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
	preprocessor.set_target_dim(2)
	preprocessor.apply_to_feature_matrix(features)
	X=preprocessor.get_transformation_matrix()
	X2=preprocessor.apply_to_feature_matrix(features)

	print 'apply to feature matrix=%d, ',n.shape(X2)
	p.plot(X2[0][:],X2[1][:],'o')
	p.title('apply_to_feature_matrix')
	p.show()
	print 'type of features=%',(type(X))
	print 'le X=\n',len(X[0])
	l1=len(X)
	l2=len(X[0])
	a1=len(data)
	a2=len(data[0])
	data2=[[0 for x in xrange(len(data))] for x in xrange(len(data[0]))]
	mulmat=[[0 for x in xrange(len(data2[0]))] for x in xrange(len(X))]
	print 'initial size of mulmat =',n.shape(mulmat)
	for i in range(a2):
		for j in range(a1):
			data2[i][j]=data[j][i]
	print 'data2(transpose of data) size =',n.shape(data2)
	print 'size of the original data is= ,and the returned matrix is',n.shape(data),n.shape(X2)
	lx0=len(X2)
	lx1=len(X2[0])
	modified_d1=[[0 for x in xrange(len(d[0]))] for x in xrange(lx0)]
	modified_d2=[[0 for x in xrange(int(len(d2[0])))] for x in xrange(lx0)]
	for i in range(lx0):
		for j in range(len(d[0])):
			modified_d1[i][j]=X2[i][j]
	for i in range(lx0):
		for j in range(lx1-len(d[0])):
			modified_d2[i][j]=X2[i][j+len(d[0])]		
	p.plot(modified_d1[0][:],modified_d1[1][:],'o',modified_d2[0][:],modified_d2[1][:],'x')
	p.title('final data')
	p.show()	
				
	return features



if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca_modular(*parameter_list[0])

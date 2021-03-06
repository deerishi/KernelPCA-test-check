
import math as m
import sys
import numpy as n
from numpy import *
import matplotlib.pyplot as p
import circle_data as cir

d=[[0 for x in xrange(50)] for x in xrange(2)] 
d2=[[0 for x in xrange(50)] for x in xrange(2)] 
d=cir.circle1
d2=cir.circle2
p.plot(d[1][:],d[0][:],'x',d2[1][:],d2[0][:],'o')
p.title('input data')
p.show()
data=hstack((d,d2))

parameter_list = [[data,0.01,1.0], [data,0.05,2.0]]
def preprocessor_kernelpca_modular (data, threshold, width):
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import KernelPCA
	from shogun.Kernel import GaussianKernel
	features = RealFeatures(data)
	kernel=GaussianKernel(features,features,width)
	preprocessor=KernelPCA(kernel)
	preprocessor.init(features)
	preprocessor.set_target_dim(2)
	#X=preprocessor.get_transformation_matrix()
	X2=preprocessor.apply_to_feature_matrix(features)
	print 'apply to feature matrix=%d, ',n.shape(X2)
	p.plot(X2,'o')
	p.title('apply_to_feature_matrix')
	p.show()
	#sys.exit(0)
	print 'type of features=%',(type(X2))
	print 'le X=\n',len(X2[0])
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
	print 'size of new datasets are =',n.shape(d),n.shape(d2)
	p.plot(modified_d1[0][:],modified_d1[1][:],'o',modified_d2[0][:],modified_d2[1][:],'x')
	p.title('final data')
	p.show()	
	print 'new d1=',modified_d1			
	return features

if __name__=='__main__':
	print('KernelPCA')
	preprocessor_kernelpca_modular(*parameter_list[0])

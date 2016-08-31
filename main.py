import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import get_feature

names= ['/home/jhm/mxnet/example/english/letter/'+line.split()[0] for line in open('/home/jhm/mxnet/example/english/letter/image_list.txt').readlines()]
l=len(names)
#datatset label
target=[]
for i in range(l):
    name=names[i].split('/')[-1]
    target.append(name[0])
    
target_ascll=[]
for i in range(l):
    target_ascll.append(ord(target[i]))
    
 
dataset=ones([l,1024])
for i in range(len(names)):
    n=get_feature.getfeature(names[i])
    dataset[i,:]=n.reshape((1, ) + n.shape)

#luanxu
np.random.seed(285)
p=np.random.permutation(dataset.shape[0])
X=ones([l,1024])
Y=ones([l])
for i in range(l):
   X[i,]=dataset[(p[i]),]
   Y[i]=target_ascll[(p[i])]
   
#train and test
X=X.astype(np.float32)
Y=Y.astype(np.float32)
data_train, data_test, target_train, target_test = train_test_split(X, Y,test_size=0.1
)


clf = SVC()
clf.fit(data_train, target_train) 
print clf.score(data_test, target_test)

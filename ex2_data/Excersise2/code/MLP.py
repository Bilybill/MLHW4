#!/usr/bin/env python
# coding: utf-8

# # Data process

# In[47]:


import numpy as np
from sklearn.model_selection import KFold

class Dataprocess:
    def getAugmentedData(self,data):
        _,cols = data.shape
        ones = np.ones((1,cols))
        data = np.vstack((ones,data))
        return data
    def __init__(self,data_path,label_path=None):
        if label_path != None:
            with open(data_path) as f:
                self.data = np.loadtxt(f,str,skiprows=1,delimiter = ",")[:,1:].astype(np.float64)
                self.data = self.data.T
            with open(label_path) as f:
                label = np.loadtxt(f,str,skiprows=1,delimiter = ",",usecols=(1,))[np.newaxis,:]
                _,cols = label.shape
                self.label = np.ones((1,cols))
                lag = np.unique(label)[1]
                self.label[label == lag] = 0
                self.label = self.label.T
        else:
            self.data = data_path[0]
            self.label = data_path[1]
    def getCrossValidationData(self,k):
        kf = KFold(n_splits=k,shuffle=True)
        subData_train = []
        subData_test = []
        for train_index, test_index in kf.split(self.data):
            #print('train_index', train_index, 'test_index', test_index)
            train_X, train_y = self.data[train_index], self.label[train_index]
            test_X, test_y = self.data[test_index], self.label[test_index]
            #print("trainx shape",train_X.shape,"testx shape",test_X.shape)
            subdata1 = Dataprocess([train_X,train_y])
            subdata2 = Dataprocess([test_X,test_y])
            subData_train.append(subdata1)
            subData_test.append(subdata2)
        return subData_train,subData_test


# # Loss Function

# In[48]:


class Lossfunction:
    def __init__(self,name):
        self.name = name
    def forward(self,output,label,categoty=2):
        if self.name == "square_loss":
            self.loss = output - label
            return np.sum(self.loss*self.loss)/self.loss.shape[0]/2
        elif self.name == "cross_entropy":
            epsi = 1e-10
            self.output = output
            self.label = label
            self.loss = label*self.output + (1-label)*(1-self.output)
            return -np.sum(np.log(self.loss+epsi))/self.loss.shape[0]
        else:
            raise ValueError("Loss function should be square loss or cross entropy")
    
    def backward(self):
        if self.name == "square_loss":
            return (self.loss)/self.loss.shape[0]
        elif self.name == "cross_entropy":
            epsi=1e-8
            return -(self.label/(self.output+epsi)+(1-self.label)/(self.output-1+epsi))/self.loss.shape[0]


# # Activation function

# In[49]:


class activationFunc:
    def __init__(self,name):
        self.name = name
    def forward(self,datain):
        self.in_data = datain
        if self.name == "sigmoid":
            self.out_data = 1/(1+np.exp(-self.in_data))
            return self.out_data
        elif self.name == "relu":
            self.in_data[self.in_data<0]=0
            return self.in_data
        else:
            raise ValueError("activation function should be relu or sigmoid")
    def backward(self):
        if self.name == "relu":
            gradient = np.zeros(self.in_data.shape)
            gradient[self.in_data>0]=1
            return gradient
        elif self.name == "sigmoid":
            return self.out_data*(1-self.out_data)
        else:
            raise ValueError("activation function should be relu or sigmoid")


# # Layer class

# In[50]:


class Layer:
    def __init__(self,shape,activation_func,lr):
        self.w = np.random.randn(shape[0],shape[1])
        self.activation = activationFunc(activation_func)
        self.lr = lr
    def forward(self,in_data):
        self.bottom_val = in_data
        return self.activation.forward(np.dot(in_data,self.w))
    def backward(self,loss):
        residual = loss*self.activation.backward()
        grad_w = np.dot(self.bottom_val.T,residual)
        self.w -= self.lr*grad_w
        residual_x = np.dot(residual,self.w.T)
        return residual_x


# # History class
# ## to record train process

# In[51]:


import matplotlib
import matplotlib.pyplot as plt

class History:
    def __init__(self):
        self.acc = []
        self.loss = []
    def addHis(self,acc,loss):
        self.acc.append(acc)
        self.loss.append(loss)
    def plotHis(self,name=None):
        plt.plot(self.acc)
        plt.plot(self.loss)
        plt.title("model accuracy and loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy and Loss")
        plt.legend(['accuracy','loss'],loc="upper left")
        if name!= None:
            plt.savefig(name)
        plt.show()


# # MLP

# In[98]:


class MLP:
    def __init__(self,loss_func,lr):
        self.lr = lr
        self.layers = []
        self.lossfunction = Lossfunction(loss_func)
        
    def addLayer(self,shape,activation_func):
        layer = Layer(shape,activation_func,self.lr)
        self.layers.append(layer)
        
    def compute_acc(self,output,label):
        lag = output.copy()
        lag[lag>0.5] = 1
        lag[lag<=0.5] = 0
        return np.sum(lag==label)/lag.shape[0]
    
    def _forward(self,in_data,in_label):
        layer_out = in_data
        for layer in self.layers:
            layer_out = layer.forward(layer_out)
        loss = self.lossfunction.forward(layer_out,in_label)
        return layer_out,loss
    def _backward(self):
        layer_loss = self.lossfunction.backward()
        for layer in self.layers[::-1]:
            layer_loss = layer.backward(layer_loss)
            
    def train(self,data,iterations,batch_size,verbose=True,val_data=None):
        his = History()
        in_data = data.data
        in_label = data.label
        batch_number = in_data.shape[0]//batch_size
        rem_number = in_data.shape[0]%batch_size
        val_bestacc = -np.inf
        flag = 0
        for i in range(iterations):
            for j in range(batch_number):
                train_data = in_data[j*batch_size:(j+1)*batch_size,:]
                train_label = in_label[j*batch_size:(j+1)*batch_size,:]
                layer_out,loss = self._forward(train_data,train_label)
                self._backward()
                acc = self.compute_acc(layer_out,train_label)
            if rem_number != 0:
                train_data = in_data[batch_number*batch_size:,:]
                train_label = in_label[batch_number*batch_size:,:]
                layer_out,loss = self._forward(train_data,train_label)
                self._backward()
                acc = self.compute_acc(layer_out,train_label)
            if verbose:
                print("epoch:%i   loss:%lf   accuracy:%lf   error:%lf"%(i,loss,acc,1-acc))
            his.addHis(acc,loss)
            if val_data!=None:
                val_output,val_loss = self._forward(val_data.data,val_data.label)
                val_acc = self.compute_acc(val_output,val_data.label)
                if val_bestacc <= val_acc:
                    val_bestacc = val_acc
                    flag = 0
                else:
                    flag += 1
            if flag == 400:
                break
        final_output,final_loss = self._forward(in_data,in_label)
        final_acc = self.compute_acc(final_output,in_label)
        #print("val_bestacc",val_bestacc)
        #print("final_acc={0},finanl_error={1},final_loss={2}".format(final_acc,1-final_acc,final_loss))
        return his
    
    def score(self,data):
        output,loss = self._forward(data.data,data.label)
        acc = self.compute_acc(output,data.label)
        print("accuracy",acc,"loss:",loss)
        return acc,loss

if __name__ == "__main__":
    trainingset_1 = Dataprocess("train_10gene_sub.csv","train_10gene_label_sub.csv")
    trainingset_2 = Dataprocess("train_10gene.csv","train_label.csv")
    testset = Dataprocess("test_10gene.csv","test_label.csv")
    testset_2 = Dataprocess("test2_10gene.csv","test2_label.csv")


# # Prepare to train and test

if __name__ == "__main__":
    mlp = MLP(loss_func="cross_entropy",lr=0.01)
    mlp.addLayer(shape=(10,20),activation_func="sigmoid")
    mlp.addLayer(shape=(20,1),activation_func="sigmoid")
    his = mlp.train(trainingset_1,batch_size = 64,iterations=3000,val_data = testset)
    print(mlp.score(testset))
    print(mlp.score(testset_2))
    his.plotHis("t2trainerror.png")


# ## cross validation

# In[107]:


if __name__ == "__main__":
    k=10
    Cro_val_traindata,Cro_val_testdata = trainingset_1.getCrossValidationData(k)
    v1 = []
    for i in range(k):
        mlp = MLP(loss_func="cross_entropy",lr=0.1)
        mlp.addLayer(shape=(10,20),activation_func="sigmoid")
        mlp.addLayer(shape=(20,1),activation_func="sigmoid")
        his = mlp.train(Cro_val_traindata[i],batch_size = 100,iterations=1000,verbose=False)
        v = mlp.score(Cro_val_testdata[i])
        v1.append(v[0])
    v1 = np.array(v1)
    print("mean acc",np.mean(v1))

# In[99]:





# ## training error

# In[106]:


if __name__ == "__main__":
    mlp = MLP(loss_func="cross_entropy",lr=0.01)
    mlp.addLayer(shape=(10,20),activation_func="sigmoid")
    mlp.addLayer(shape=(20,1),activation_func="sigmoid")
    his = mlp.train(trainingset_2,batch_size = 64,iterations=3000,val_data = testset)
    print(mlp.score(testset))
    print(mlp.score(testset_2))
    his.plotHis("t2trainerror.png")


# ## cross validation

# In[107]:


if __name__ == "__main__":
    k=10
    Cro_val_traindata,Cro_val_testdata = trainingset_2.getCrossValidationData(k)
    v1 = []
    for i in range(k):
        mlp = MLP(loss_func="cross_entropy",lr=0.1)
        mlp.addLayer(shape=(10,20),activation_func="sigmoid")
        mlp.addLayer(shape=(20,1),activation_func="sigmoid")
        his = mlp.train(Cro_val_traindata[i],batch_size = 100,iterations=1000,verbose=False)
        v = mlp.score(Cro_val_testdata[i])
        v1.append(v[0])
    v1 = np.array(v1)
    print("mean acc",np.mean(v1))


# In[28]:


if __name__ == "__main__":
    his.plotHis()






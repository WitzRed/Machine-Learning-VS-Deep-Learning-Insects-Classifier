#Fastai course
#%%
import os
from PIL import Image
from array import *
from random import shuffle
#%%
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gs
# In[2]:

from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *


# In[3]:

#path = 'data/mnist/'
data_path = ".\\insects\\"

#%%
train_data_dir = os.path.join(data_path,'train')
validadtion_data_dir = os.path.join(data_path,'valid')
test_data_dir = os.path.join(data_path,'test')
#%%
class_names = os.listdir(train_data_dir) #get name of classes

#%%
#Prepare names
Names = [[train_data_dir,'train'],[validadtion_data_dir,'valid'],[test_data_dir,'test']]
lX_train = []
ly_train = []
    
lX_valid = []
ly_valid = []
    
lX_test = []
ly_test = []
#%%
#Start pre-processing... 
#prepare the dataset Serializing the images and and turning their folders into labels.
for name in Names:
    print('name: ' + str(name))
    FileList = []
    for dirname in os.listdir(name[0]):
        print('dirname: '+str(dirname))
        path =  os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                FileList.append(os.path.join(name[0],dirname,filename))
                
    
    shuffle(FileList)
    record = int(0)
    for filename in FileList:
        set_string= filename.split('\\')[2]
        label_string= filename.split('\\')[3]  
        if record ==0:
            print(set_string)
            print(label_string)
            record = 1
        
        if set_string == 'train':           
            Im = Image.open(filename)
            Im = Im.resize((224,224))
            im_array = np.array(Im)
            lX_train.append(im_array.reshape(224*224*3))           
            ly_train.append(int(class_names.index(label_string)))
            
        if set_string == 'valid':           
            Im = Image.open(filename)
            Im = Im.resize((224,224))
            im_array = np.array(Im)
            lX_valid.append(im_array.reshape(224*224*3))          
            ly_valid.append(int(class_names.index(label_string)))
            
        if set_string == 'test':           
            Im = Image.open(filename)
            Im = Im.resize((224,224))
            im_array = np.array(Im)
            lX_test.append(im_array.reshape(224*224*3))         
            ly_test.append(int(class_names.index(label_string)))
            

# In[7]: 
#Convert Lists to numpy arrays
X_train = np.array(lX_train, dtype=np.int16)
y_train = np.array(ly_train, dtype=np.int16)
    
X_valid = np.array(lX_valid, dtype=np.int16)
y_valid = np.array(ly_valid , dtype=np.int16)
    
X_test = np.array(lX_test, dtype=np.int16)
y_test = np.array(ly_test, dtype=np.int16)

#delete unecessary variables
del(lX_train, ly_train, lX_valid, ly_valid, lX_test, ly_test,FileList)

#%%
os.listdir(data_path)


# In[10]:
#checking subfolders or class names

os.listdir(f'{data_path}valid')


# In[11]:


files = os.listdir(f'{data_path}valid/{class_names[0]}')[:5]
files


# In[12]:
#Show an image

img = plt.imread(f'{data_path}valid/{class_names[0]}/{files[0]}')
plt.imshow(img);


# Here is how the raw data looks like

# In[11]:
#Display raw data dimension

img.shape


# In[12]:
#Image pixels samples

img[:4,:4]

#%% Check data types and shapes
type(X_train), X_train.shape, type(y_train ), y_train .shape

#%%
#Visualizing New Normalized data
x_imgs = np.reshape(X_valid,(-1,224,224,3));x_imgs.shape

plt.imshow(x_imgs[0])

#%% Normalize
mean = X_train.mean()
std = X_train.std()

#%%

X_train = (X_train-mean)/std
mean, std, X_train.mean(), X_train.std()

#%%
X_valid = (X_valid-mean)/std
X_valid.mean(), X_valid.std()

#%%
#Visualizing New Normalized data
x_imgs_norm = np.reshape(X_valid,(-1,224,224,3));x_imgs_norm.shape

plt.imshow(x_imgs_norm[0])

del(x_imgs_norm)

# In[36]:

def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


# In[37]:

def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


# #### Plots 

# In[38]:

X_valid.shape


# In[40]:

show(x_imgs[1], y_valid[1])


# In[41]:

y_valid.shape


# It's the digit 3!  And that's stored in the y value:

# In[42]:

y_valid[0]


# We can look at part of an image:

# In[19]:

x_imgs[0,10:15,10:15]


# In[20]:

show(x_imgs[0,10:15,10:15])


# In[21]:

plots(x_imgs[:8], titles=[class_names[y_valid[0]],class_names[y_valid[1]],
                          class_names[y_valid[2]],class_names[y_valid[3]],
                          class_names[y_valid[4]],class_names[y_valid[5]],
                          class_names[y_valid[6]],class_names[y_valid[7]]])

# In[14]:

from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *
from fastai.plots import *

import torch.nn as nn


# We will begin with the highest level abstraction: using a neural net defined by PyTorch's Sequential class.  
#%%
# Size of images when they are used by GPU
sze=224
bsz = 16

# In[15]: 
#Neural Network Architecture
net = nn.Sequential(
        nn.Linear(sze*sze*3,len(class_names)),
        nn.LogSoftmax()
        ).cuda()

#%% Define our Model data
md = ImageClassifierData.from_arrays(data_path, (X_train,y_train), (X_valid, y_valid))


# In[17]:

loss=nn.NLLLoss()
metrics=[accuracy]
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)
opt=optim.SGD(net.parameters(), 6e-2, momentum=0.7, weight_decay=2e-3)

#%% Start Laerning
fit(net, md, n_epochs=92, crit=loss, opt=opt, metrics=metrics)

# ### Loss functions and metrics

# In machine learning the **loss** function or cost function is representing the price paid for inaccuracy of predictions.
# 
# The loss associated with one example in binary classification is given by:
# `-(y * log(p) + (1-y) * log (1-p))`
# where `y` is the true label of `x` and `p` is the probability predicted by our model that the label is 1.

# In[15]:

def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))


# In[16]:

acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)


# Note that in our toy example above our accuracy is 100% and our loss is 0.16. Compare that to a loss of 0.03 that we are getting while predicting cats and dogs. Exercise: play with `preds` to get a lower loss for this example. 
# 
# **Example:** Here is an example on how to compute the loss for one example of binary classification problem. Suppose for an image x with label 1 and your model gives it a prediction of 0.9. For this case the loss should be small because our model is predicting a label $1$ with high probability.
# 
# `loss = -log(0.9) = 0.10`
# 
# Now suppose x has label 0 but our model is predicting 0.9. In this case our loss is should be much larger.
# 
# `loss = -log(1-0.9) = 2.30`
# 
# - Exercise: look at the other cases and convince yourself that this make sense.
# - Exercise: how would you rewrite `binary_loss` using `if` instead of `*` and `+`?
# 
# Why not just maximize accuracy? The binary classification loss is an easier function to optimize.
# 
# For multi-class classification, we use *negative log liklihood* (also known as *categorical cross entropy*) which is exactly the same thing, but summed up over all classes.

# ### Fitting the model

# *Fitting* is the process by which the neural net learns the best parameters for the dataset.
#%% Change the Optimizer to Adam
opt=optim.Adam(net.parameters())

# In[127]:

fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)


# In[128]:

set_lrs(opt, 5e-2)


# In[129]:

fit(net, md, n_epochs=32, crit=loss, opt=opt, metrics=metrics)


# In[ ]:
net = nn.Sequential(
    nn.Linear(sze*sze*3, len(class_names)),
    nn.ReLU(),
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, len(class_names)),
    nn.LogSoftmax()
).cuda()

#%% Change the Optimizer to Adam
opt=optim.Adam(net.parameters())


# In[133]:

fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)


# In[134]:

set_lrs(opt, 1e-2)


# In[135]:

fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)


# In[100]:

t = [o.numel() for o in net.parameters()]
t, sum(t)


# GPUs are great at handling lots of data at once (otherwise don't get performance benefit).  We break the data up into **batches**, and that specifies how many samples from our dataset we want to send to the GPU at a time.  The fastai library defaults to a batch size of 64.  On each iteration of the training loop, the error on 1 batch of data will be calculated, and the optimizer will update the parameters based on that.
# 
# An **epoch** is completed once each data sample has been used once in the training loop.
# 
# Now that we have the parameters for our model, we can make predictions on our validation set.

# In[148]:

preds = predict(net, md.val_dl)


# In[149]:

preds.shape


# **Question**: Why does our output have length 10 (for each image)?

# In[150]:

preds.argmax(axis=1)[:5]


# In[151]:

preds = preds.argmax(1)


# Let's check how accurate this approach is on our validation set. You may want to compare this against other implementations of logistic regression, such as the one in sklearn. In our testing, this simple pytorch version is faster and more accurate for this problem!

# In[152]:

np.mean(preds == y_valid)


# Let's see how some of our predictions look!

# In[153]:

plots(x_imgs[:8], titles=[class_names[preds[0]],class_names[preds[1]],
                          class_names[preds[2]],class_names[preds[3]],
                          class_names[preds[4]],class_names[preds[5]],
                          class_names[preds[6]],class_names[preds[7]]])


# ## Defining Logistic Regression Ourselves

# Above, we used pytorch's `nn.Linear` to create a linear layer.  This is defined by a matrix multiplication and then an addition (these are also called `affine transformations`).  Let's try defining this ourselves.
# 
# Just as Numpy has `np.matmul` for matrix multiplication (in Python 3, this is equivalent to the `@` operator), PyTorch has `torch.matmul`.  
# 
# Our PyTorch class needs two things: constructor (says what the parameters are) and a forward method (how to calculate a prediction using those parameters)  The method `forward` describes how the neural net converts inputs to outputs.
# 
# In PyTorch, the optimizer knows to try to optimize any attribute of type **Parameter**.
#%%   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, preds)


# We can just print out the confusion matrix, or we can show a graphical view (which is mainly useful for dependents with a larger number of categories).

# In[53]:


matplotlib.rcParams['figure.figsize'] = [15,10]


# In[54]:

plot_confusion_matrix(cm, class_names)


# In[18]:

def get_weights(*dims): return nn.Parameter(torch.randn(dims)/dims[0])
def softmax(x): return torch.exp(x)/(torch.exp(x).sum(dim=1)[:,None])

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(sze*sze*3, len(class_names))  # Layer 1 weights
        self.l1_b = get_weights(len(class_names))         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = (x @ self.l1_w) + self.l1_b  # Linear Layer
        x = torch.log(softmax(x)) # Non-linear (LogSoftmax) Layer
        return x


# We create our neural net and the optimizer.  (We will use the same loss and metrics from above).

# In[19]:

net2 = LogReg().cuda()
opt=optim.Adam(net2.parameters())


# In[20]:

fit(net2, md, n_epochs=10, crit=loss, opt=opt, metrics=metrics)


# In[21]:

dl = iter(md.trn_dl)


# In[22]:

xmb,ymb = next(dl)


# In[23]:

vxmb = Variable(xmb.cuda())
vxmb


# In[24]:

preds = net2(vxmb).exp(); preds[:3]


# In[25]:

preds = preds.data.max(1)[1]; preds


# Let's look at our predictions on the first eight images:

# In[43]:

preds = predict(net2, md.val_dl).argmax(1)
plots(x_imgs[:8], titles=preds[:8])


# In[45]:

np.mean(preds == y_valid)


# ## Aside about Broadcasting and Matrix Multiplication

# Now let's dig in to what we were doing with `torch.matmul`: matrix multiplication.  First, let's start with a simpler building block: **broadcasting**.

# ### Element-wise operations 

# Broadcasting and element-wise operations are supported in the same way by both numpy and pytorch.
# 
# Operators (+,-,\*,/,>,<,==) are usually element-wise.
# 
# Examples of element-wise operations:

# In[80]:

a = np.array([10, 6, -4])
b = np.array([2, 8, 7])
a,b


# In[81]:

a + b


# In[84]:

(a < b).mean()


# ### Broadcasting

# The term **broadcasting** describes how arrays with different shapes are treated during arithmetic operations.  The term broadcasting was first used by Numpy, although is now used in other libraries such as [Tensorflow](https://www.tensorflow.org/performance/xla/broadcasting) and Matlab; the rules can vary by library.
# 
# From the [Numpy Documentation](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html):
# 
#     The term broadcasting describes how numpy treats arrays with 
#     different shapes during arithmetic operations. Subject to certain 
#     constraints, the smaller array is “broadcast” across the larger 
#     array so that they have compatible shapes. Broadcasting provides a 
#     means of vectorizing array operations so that looping occurs in C
#     instead of Python. It does this without making needless copies of 
#     data and usually leads to efficient algorithm implementations.
#     
# In addition to the efficiency of broadcasting, it allows developers to write less code, which typically leads to fewer errors.
# 
# *This section was adapted from [Chapter 4](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression) of the fast.ai [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra) course.*

# #### Broadcasting with a scalar

# In[85]:

a


# In[86]:

a > 0


# How are we able to do a > 0?  0 is being **broadcast** to have the same dimensions as a.
# 
# Remember above when we normalized our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar)?  We were using broadcasting!
# 
# Other examples of broadcasting with a scalar:

# In[165]:

a + 1


# In[91]:

m = np.array([[1, 2, 3], [4,5,6], [7,8,9]]); m


# In[92]:

2*m


# #### Broadcasting a vector to a matrix

# We can also broadcast a vector to a matrix:

# In[93]:

c = np.array([10,20,30]); c


# In[94]:

m + c


# In[95]:

c + m


# Although numpy does this automatically, you can also use the `broadcast_to` method:

# In[96]:

c.shape


# In[111]:

np.broadcast_to(c[:,None], m.shape)


# In[98]:

np.broadcast_to(np.expand_dims(c,0), (3,3))


# In[99]:

c.shape


# In[100]:

np.expand_dims(c,0).shape


# The numpy `expand_dims` method lets us convert the 1-dimensional array `c` into a 2-dimensional array (although one of those dimensions has value 1).

# In[172]:

np.expand_dims(c,0).shape


# In[173]:

m + np.expand_dims(c,0)


# In[103]:

np.expand_dims(c,1)


# In[109]:

c[:, None].shape


# In[101]:

m + np.expand_dims(c,1)


# In[62]:

np.broadcast_to(np.expand_dims(c,1), (3,3))


# #### Broadcasting Rules

# In[114]:

c[None]


# In[115]:

c[:,None]


# In[118]:

c[None] > c[:,None]


# In[123]:

xg,yg = np.ogrid[0:5, 0:5]; xg,yg


# In[122]:

xg+yg


# When operating on two arrays, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when
# 
# - they are equal, or
# - one of them is 1
# 
# Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:
# 
#     Image  (3d array): 256 x 256 x 3
#     Scale  (1d array):             3
#     Result (3d array): 256 x 256 x 3
# 
# The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.

# ### Matrix Multiplication

# We are going to use broadcasting to define matrix multiplication.

# In[124]:

m, c


# In[105]:

m @ c  # np.matmul(m, c)


# We get the same answer using `torch.matmul`:

# In[125]:

T(m) @ T(c)


# The following is **NOT** matrix multiplication.  What is it?

# In[108]:

m,c


# In[179]:

m * c


# In[180]:

(m * c).sum(axis=1)


# In[181]:

c


# In[182]:

np.broadcast_to(c, (3,3))


# From a machine learning perspective, matrix multiplication is a way of creating features by saying how much we want to weight each input column.  **Different features are different weighted averages of the input columns**. 
# 
# The website [matrixmultiplication.xyz](http://matrixmultiplication.xyz/) provides a nice visualization of matrix multiplcation

# In[109]:

n = np.array([[10,40],[20,0],[30,-5]]); n


# In[110]:

m


# In[71]:

m @ n


# In[184]:

(m * n[:,0]).sum(axis=1)


# In[185]:

(m * n[:,1]).sum(axis=1)


# ## Writing Our Own Training Loop

# As a reminder, this is what we did above to write our own logistic regression class (as a pytorch neural net):

# In[26]:

# Our code from above
class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x @ self.l1_w + self.l1_b 
        return torch.log(softmax(x))

net2 = LogReg().cuda()
opt=optim.Adam(net2.parameters())

fit(net2, md, n_epochs=1, crit=loss, opt=opt, metrics=metrics)


# Above, we are using the fastai method `fit` to train our model.  Now we will try writing the training loop ourselves.
# 
# **Review question:** What does it mean to train a model?

# We will use the LogReg class we created, as well as the same loss function, learning rate, and optimizer as before:

# In[27]:

net2 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-3
optimizer=optim.Adam(net2.parameters(), lr=learning_rate)


# md is the ImageClassifierData object we created above.  We want an iterable version of our training data (**question**: what does it mean for something to be iterable?):

# In[28]:

dl = iter(md.trn_dl) # Data loader


# First, we will do a **forward pass**, which means computing the predicted y by passing x to the model.

# In[29]:

xt, yt = next(dl)
y_pred = net2(Variable(xt).cuda())


# We can check the loss:

# In[30]:

l = loss(y_pred, Variable(yt).cuda())
print(l)


# We may also be interested in the accuracy.  We don't expect our first predictions to be very good, because the weights of our network were initialized to random values.  Our goal is to see the loss decrease (and the accuracy increase) as we train the network:

# In[31]:

np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))


# Now we will use the optimizer to calculate which direction to step in.  That is, how should we update our weights to try to decrease the loss?
# 
# Pytorch has an automatic differentiation package ([autograd](http://pytorch.org/docs/master/autograd.html)) that takes derivatives for us, so we don't have to calculate the derivative ourselves!  We just call `.backward()` on our loss to calculate the direction of steepest descent (the direction to lower the loss the most).

# In[32]:

# Before the backward pass, use the optimizer object to zero all of the
# gradients for the variables it will update (which are the learnable weights
# of the model)
optimizer.zero_grad()

# Backward pass: compute gradient of the loss with respect to model parameters
l.backward()

# Calling the step function on an Optimizer makes an update to its parameters
optimizer.step()


# Now, let's make another set of predictions and check if our loss is lower:

# In[33]:

xt, yt = next(dl)
y_pred = net2(Variable(xt).cuda())


# In[34]:

l = loss(y_pred, Variable(yt).cuda())
print(l)


# Note that we are using **stochastic** gradient descent, so the loss is not guaranteed to be strictly better each time.  The stochasticity comes from the fact that we are using **mini-batches**; we are just using 64 images to calculate our prediction and update the weights, not the whole dataset.

# In[35]:

np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))


# If we run several iterations in a loop, we should see the loss decrease and the accuracy increase with time.

# In[36]:

for t in range(100):
    xt, yt = next(dl)
    y_pred = net2(Variable(xt).cuda())
    l = loss(y_pred, Variable(yt).cuda())
    
    if t % 10 == 0:
        accuracy = np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))
        print("loss: ", l.data[0], "\t accuracy: ", accuracy)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()


# ### Put it all together in a training loop

# In[37]:

def score(x, y):
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)


# In[38]:

net2 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-2
optimizer=optim.SGD(net2.parameters(), lr=learning_rate)

for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        # Forward pass: compute predicted y and loss by passing x to the model.
        xt, yt = next(dl)
        y_pred = net2(V(xt))
        l = loss(y_pred, V(yt))
        losses.append(l)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    
    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))


# ## Stochastic Gradient Descent

# Nearly all of deep learning is powered by one very important algorithm: **stochastic gradient descent (SGD)**. SGD can be seeing as an approximation of **gradient descent (GD)**. In GD you have to run through all the samples in your training set to do a single itaration. In SGD you use only a subset of training samples to do the update for a parameter in a particular iteration. The subset used in each iteration is called a batch or minibatch.
# 
# Now, instead of using the optimizer, we will do the optimization ourselves!

# In[39]:

net2 = LogReg().cuda()
loss_fn=nn.NLLLoss()
lr = 1e-2
w,b = net2.l1_w,net2.l1_b

for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        xt, yt = next(dl)
        y_pred = net2(V(xt))
        l = loss(y_pred, Variable(yt).cuda())
        losses.append(loss)

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()
        w.data -= w.grad.data * lr
        b.data -= b.grad.data * lr
        
        w.grad.data.zero_()
        b.grad.data.zero_()   

    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))


# In[ ]:




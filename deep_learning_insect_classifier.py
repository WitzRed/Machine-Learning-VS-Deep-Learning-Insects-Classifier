
# coding: utf-8

# # Estudo de caso: reconhecimento de insetos
# Witenberg S R Souza - 30/10/2018

# In[23]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


from fastai.conv_learner import *

#%%

import matplotlib.pyplot as plt
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

# ## Setup the path to data

# In[25]:


# Put the path to your senators folder that corresponds to your installation
PATH = "C:/Users/Witenberg/Documents/DEEP_LEARNING_TCU/fastai/courses/dl1/data/insects/"
#%%
#Set class names
class_names = os.listdir(f'{PATH}valid')
        
# In[5]:    
# Size of images when they are used by GPU
sz=224
bsz = 16
#%%


torch.cuda.is_available()


# In addition, NVidia provides special accelerated functions for deep learning in a package called CuDNN. Although not strictly necessary, it will improve training performance significantly, and is included by default in all supported fastai configurations. Therefore, if the following does not return `True`, you may want to look into why.

# In[6]:


torch.backends.cudnn.enabled
# In[26]:


# Uncomment the below if you need to reset your precomputed activations
#shutil.rmtree(f'{PATH}tmp', ignore_errors=True)

#In the lesson1.ipynb, Jeremy Howard uses resnet34.

#After completing this exercice with resnet34, test resnet50, resnet101 and resnet152. Check if the validation accuracy changes. 

# our model
# If you want to know the list of the pretrained models under pytorch : http://pytorch.org/docs/master/torchvision/models.html

arch=resnet34
#arch=resnet50
# arch=resnet101
# arch=resnet152

# our data transformation
#tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
# learn = ...
data = ImageClassifierData.from_paths(PATH, bs=bsz, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.1, 3)


# In[27]:
learn.fit(1e-2, 32)

#%%

# this gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()
log_preds.shape
preds = np.argmax(log_preds,axis=1)
log_preds[:5]


# In[28]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[29]:


lrf=learn.lr_find()


# Our `learn` object contains an attribute `sched` that contains our learning rate scheduler, and has some convenient plotting functionality including this one:

# In[30]:


learn.sched.plot_lr()


# Note that in the previous plot *iteration* is one iteration (or *minibatch*) of SGD. In one epoch there are 
# (num_train_samples/num_iterations) of SGD.
# 
# We can see the plot of loss versus learning rate to see where our loss stops decreasing:

# In[31]:


learn.sched.plot()


# ## Improving our model

# ### Data augmentation

# If you try training for more epochs, you'll notice that we start to *overfit*, which means that our model is learning to recognize the specific images in the training set, rather than generalizaing such that we also get good results on the validation set. One way to fix this is to effectively create more data, through *data augmentation*. This refers to randomly changing the images in ways that shouldn't impact their interpretation, such as horizontal flipping, zooming, and rotating.
# 
# We can do this by passing `aug_tfms` (*augmentation transforms*) to `tfms_from_model`, with a list of functions to apply that randomly change the image however we wish. For photos that are largely taken from the side (e.g. most photos of dogs and cats, as opposed to photos taken from the top down, such as satellite imagery) we can use the pre-defined list of functions `transforms_side_on`. We can also specify random zooming of images up to specified scale by adding the `max_zoom` parameter.

# In[32]:


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


# In[33]:


def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x,_ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]


# In[34]:


data = ImageClassifierData.from_paths(PATH, bs=bsz,tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[35]:


learn.fit(1e-3, 10)


# In[36]:


learn.precompute=False


# By default when we create a learner, it sets all but the last layer to *frozen*. That means that it's still only updating the weights in the last layer when we call `fit`.

# In[37]:


learn.fit(1e-3, 5, cycle_len=1)


# In[38]:


data = ImageClassifierData.from_paths(PATH, bs=bsz,tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[39]:


learn.fit(5e-4,10)


# In[40]:


learn.precompute=False


# By default when we create a learner, it sets all but the last layer to *frozen*. That means that it's still only updating the weights in the last layer when we call `fit`.

# In[41]:


learn.fit(5e-4, 3, cycle_len=1)


# ### Fine-tuning and differential learning rate annealing

# Now that we have a good final layer trained, we can try fine-tuning the other layers. To tell the learner that we want to unfreeze the remaining layers, just call (surprise surprise!) `unfreeze()`.

# In[42]:


learn.unfreeze()


# Note that the other layers have *already* been trained to recognize imagenet photos (whereas our final layers where randomly initialized), so we want to be careful of not destroying the carefully tuned weights that are already there.
# 
# Generally speaking, the earlier layers (as we've seen) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2 as before. We refer to this as *differential learning rates*, although there's no standard name for this techique in the literature that we're aware of.

# In[43]:


lr=np.array([5e-4,1e-3,1e-2])


# In[44]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# Another trick we've used here is adding the `cycle_mult` parameter. Take a look at the following chart, and see if you can figure out what the parameter is doing:

# In[45]:


learn.sched.plot_lr()


# In[46]:


log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# In[47]:


accuracy_np(probs, y)


# I generally see about a 10-20% reduction in error on this dataset when using TTA at this point, which is an amazing result for such a quick and easy technique!

# ## Analyzing results

# ### Confusion matrix 

# In[51]:


preds = np.argmax(probs, axis=1)
probs = probs[:,1]


# A common way to analyze the result of a classification model is to use a [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/). Scikit-learn has a convenient function we can use for this purpose:
#%%
#Plot some results

def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)
def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y)==is_correct)


# In[20]:


def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)


# In[21]:


def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


# In[22]:


def load_img_id(ds, idx): return np.array(PIL.Image.open(PATH+ds.fnames[idx]))

def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds,x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16,8))


# In[23]:


# 1. A few correct labels at random
plot_val_with_title(rand_by_correct(True), "Correctly classified")

# In[52]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds)



# We can just print out the confusion matrix, or we can show a graphical view (which is mainly useful for dependents with a larger number of categories).

# In[53]:


matplotlib.rcParams['figure.figsize'] = [15,10]


# In[54]:

plot_confusion_matrix(cm, data.classes)


# In[ ]:


learn.save('insects_224_lastlayer')


# In[ ]:


learn.load('insects_224_lastlayer')


#%%
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
#%%
    
files = os.listdir(f'{PATH}test/thrips')[:]
#%% Load a single image
#Test
img1 = plt.imread(f'{PATH}test/thrips/{files[5]}')
plt.imshow(img1);
#%%
trn_tfms, val_tfms = tfms_from_model(arch,sz)#get transformations
im = val_tfms(open_image(f'{PATH}test/thrips/{files[5]}'))
learn.precompute = False
prediction = learn.predict_array(im[None])
preds = np.argmax(prediction)
print (preds)
print('O inseto é da classe '+str(data.classes[preds])+' : '+str(preds)+", sem biologismo, por favor.")


#%%
imgs =[]
test_preds=[]
trn_tfms, val_tfms = tfms_from_model(arch,sz)#get transformations
for insects in data.classes:
    print(insects)
    files = os.listdir(f'{PATH}test/{insects}')[:5]
    img = val_tfms(open_image(f'{PATH}test/{insects}/{files[3]}'))
    prediction = learn.predict_array(img[None])
    test_preds.append(np.argmax(prediction))
    imgs.append(plt.imread(f'{PATH}test/{insects}/{files[3]}'))
    plt.imshow(imgs[0])
imgs.append(img1)
#%%
def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')

#%%
plots(imgs[:8], titles=[class_names[test_preds[0]],class_names[test_preds[1]],
                          class_names[test_preds[2]],class_names[test_preds[3]],
                          class_names[test_preds[4]],class_names[test_preds[5]],
                          class_names[test_preds[6]],str(data.classes[preds])])        
        
        
#%% #Predict on test set This step is under construction...
acc =0
test_log_preds= np.zeros([7,60,7])
preds_test = np.zeros([7,60,7])
for insects in data.classes:
    data = ImageClassifierData.from_paths(PATH, bs=bsz, tfms=tfms_from_model(arch, sz), test_name = 'test/'+insects)
    learn = ConvLearner.pretrained(arch, data)
    test_log_preds[[data.classes.index(insects)]]=learn.predict(is_test = True)
    preds_test[[data.classes.index(insects)]] = np.argmax(test_log_preds[[data.classes.index(insects)]], axis = 1)
    acc = sum(preds_test[[data.classes.index(insects)]] == data.classes.index(insects))/(test_log_preds[[0]].shape[1]*test_log_preds[[0]].shape[2])
#%% Evaluate 

log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)

accuracy_np(probs, y)
print("Accuracy in test set: {:.4f}".format(acc))
data.files

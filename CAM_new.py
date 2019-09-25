#!/usr/bin/env python
# coding: utf-8

# In[14]:


from vis.utils import utils
from keras import activations
from keras import models
from keras.models import load_model, Model
from vis.utils import utils
from matplotlib import pyplot as plt
from keras.preprocessing import image
from vis.visualization import visualize_saliency, overlay, visualize_activation, visualize_cam
import numpy as np
import scipy.misc
import matplotlib
import math
from scipy.ndimage import convolve

import warnings
warnings.filterwarnings('ignore')


# In[2]:


model=load_model('1/weights-improvement-47-0.78.h5')

model.summary()


# In[3]:


layer_idx = utils.find_layer_idx(model, 'dense_1')
model.layers[layer_idx].activation = activations.linear
modelnew = utils.apply_modifications(model)
penultimate_layer = utils.find_layer_idx(modelnew, 'add_12')


# In[90]:


img1 = utils.load_img('1344_1_D_CC_R.png', target_size=(450, 450))
mean = img1.mean()
std = img1.std()
img1 = (img1 - mean)/std
print(img1.shape)


scipy.misc.imsave('1344_1_D_CC_R_450.png', img1)
img1_resized = img1[np.newaxis, ...]
img1_resized = img1_resized[..., np.newaxis]
print(img1_resized.shape)


# In[91]:


pred_class = np.argmax(model.predict(img1_resized))
print(pred_class)


# In[93]:


mappa=visualize_cam(modelnew, layer_idx, filter_indices=[pred_class], seed_input=img1_resized, penultimate_layer_idx=penultimate_layer)
print(mappa.shape)

plt.imsave('mappa_1344_D_color.png', mappa, cmap='seismic')


# In[94]:


immagine=utils.load_img('1344_1_D_CC_R_450.png')
mappa_color=utils.load_img('mappa_1344_D_color.png')
print(mappa_color.shape)

arrays=[immagine, immagine, immagine, immagine]
img1_reshape=np.stack(arrays, axis=2)
print(img1_reshape.shape)

    # add gradient as overlay
mappa_overlay=overlay(img1_reshape, mappa_color)


plt.imshow(mappa_overlay)
plt.show()


# In[95]:


weights=model.layers[-1].get_weights()
print(model.layers[-1].name)
print(len(weights))
print(weights[0].shape)

w=weights[0]
w=w[:, 3]
print(w.shape)

layer_output = model.layers[-5].output
print(layer_output.name)
print(layer_output.shape)


newModel = Model(model.inputs, layer_output)
outputs=newModel.predict(img1_resized)
print(outputs.shape)
outputs=np.reshape(outputs, (15, 15, 256))
print(outputs.shape)

p=np.zeros((15,15))
somma=np.zeros((15,15))
for i in range(w.shape[0]):
    p=w[i]*outputs[:, :, i]
    somma=somma+p

print(somma.shape)

plt.imshow(somma, cmap='seismic')
plt.show()


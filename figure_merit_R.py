#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import recall_score, accuracy_score, precision_score, classification_report


# In[2]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if not title:
    #    if normalize:
    #        title = 'Normalized confusion matrix'
    #    else:
    #        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    #if normalize:
    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)


# In[3]:


df_MLOR = pd.read_csv('test/predictions_mlor.txt', sep=" ", header=None)   #Read a comma-separated values (csv) file into DataFrame.
df_CCR = pd.read_csv('test/predictions_ccr.txt', sep=" ", header=None)
df_MLOR.columns = ["a", "b", "c", "d"]
df_CCR.columns = ["a", "b", "c", "d"]


# In[4]:


label = 20*[0] + 46*[1] + 75*[2] + 17* [3]


# In[5]:


df_MLOR['label'] = label
df_CCR['label'] = label


# In[6]:


df_MLOR.head()


# In[7]:


df_CCR.head()


# In[9]:


prediction_MLOR = np.argmax((df_MLOR.iloc[:,0:4]).to_numpy(), axis=1)
prediction_CCR = np.argmax((df_CCR.iloc[:,0:4]).to_numpy(), axis=1)


# In[10]:


df_MLOR['predict'] = prediction_MLOR
df_CCR['predict'] = prediction_CCR


# In[11]:


df_MLOR.head()


# In[12]:


df_CCR.head()


# In[13]:


class_names = np.array(['A','B','C','D'], dtype='<U10')


# In[14]:


plot_confusion_matrix(df_MLOR['label'], df_MLOR['predict'], classes=class_names)
plt.savefig('provaconfusion_MLOR.png', dpi=300)


# In[15]:


plot_confusion_matrix(df_CCR['label'], df_CCR['predict'], classes=class_names)
plt.savefig('provaconfusion_CCR.png', dpi=300)


# In[16]:


metrics_MLOR = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(df_MLOR['label'], df_MLOR['predict'])],
    'recall':[recall_score(df_MLOR['label'], df_MLOR['predict'], average='macro')],
    'precision':[precision_score(df_MLOR['label'], df_MLOR['predict'], average='macro')]
    },
orient = 'index')


# In[17]:


metrics_MLOR


# In[18]:


metrics_CCR = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(df_CCR['label'], df_CCR['predict'])],
    'recall':[recall_score(df_CCR['label'], df_CCR['predict'], average='macro')],
    'precision':[precision_score(df_CCR['label'], df_CCR['predict'], average='macro')]
    },
orient = 'index')


# In[19]:


metrics_CCR


# In[20]:


pred_right = (df_MLOR.iloc[:,0:4] + df_CCR.iloc[:,0:4])/2


# In[21]:


pred_right


# In[22]:


pred_right['label'] = label


# In[23]:


prediction_R = np.argmax((pred_right.iloc[:,0:4]).to_numpy(), axis=1)


# In[24]:


pred_right['predict'] = prediction_R
pred_right.head(n=21)


# In[25]:


plot_confusion_matrix(pred_right['label'], pred_right['predict'], classes=class_names)
plt.savefig('provaconfusion_right.png', dpi=300)


# In[26]:


metrics_right = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(pred_right['label'], pred_right['predict'])],
    'recall':[recall_score(pred_right['label'], pred_right['predict'], average='macro')],
    'precision':[precision_score(pred_right['label'], pred_right['predict'], average='macro')]
    },
orient = 'index')


# In[27]:


metrics_right


# In[30]:


df_MLOL = pd.read_csv('test/predictions_mlol.txt', sep=" ", header=None)   #Read a comma-separated values (csv) file into DataFrame.
df_CCL = pd.read_csv('test/predictions_ccl.txt', sep=" ", header=None)
df_MLOL.columns = ["a", "b", "c", "d"]
df_CCL.columns = ["a", "b", "c", "d"]


# In[31]:


df_MLOL['label'] = label
df_CCL['label'] = label


# In[32]:


df_MLOL.head()


# In[33]:


df_CCL.head()


# In[34]:


prediction_MLOL = np.argmax((df_MLOL.iloc[:,0:4]).to_numpy(), axis=1)
prediction_CCL = np.argmax((df_CCL.iloc[:,0:4]).to_numpy(), axis=1)


# In[35]:


df_MLOL['predict'] = prediction_MLOL
df_CCL['predict'] = prediction_CCL


# In[36]:


df_MLOL.head()


# In[37]:


df_CCL.head()


# In[38]:


plot_confusion_matrix(df_MLOL['label'], df_MLOL['predict'], classes=class_names)
plt.savefig('provaconfusion_MLOL.png', dpi=300)


# In[39]:


plot_confusion_matrix(df_CCL['label'], df_CCL['predict'], classes=class_names)
plt.savefig('provaconfusion_CCL.png', dpi=300)


# In[40]:


metrics_MLOL = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(df_MLOL['label'], df_MLOL['predict'])],
    'recall':[recall_score(df_MLOL['label'], df_MLOL['predict'], average='macro')],
    'precision':[precision_score(df_MLOL['label'], df_MLOL['predict'], average='macro')]
    },
orient = 'index')


# In[41]:


metrics_MLOL


# In[42]:


metrics_CCL = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(df_CCL['label'], df_CCL['predict'])],
    'recall':[recall_score(df_CCL['label'], df_CCL['predict'], average='macro')],
    'precision':[precision_score(df_CCL['label'], df_CCL['predict'], average='macro')]
    },
orient = 'index')


# In[43]:


metrics_CCL


# In[44]:


pred_left = (df_MLOL.iloc[:,0:4] + df_CCL.iloc[:,0:4])/2


# In[45]:


pred_left


# In[46]:


pred_left['label'] = label


# In[47]:


prediction_L = np.argmax((pred_left.iloc[:,0:4]).to_numpy(), axis=1)


# In[48]:


pred_left['predict'] = prediction_L
pred_left.head()


# In[49]:


plot_confusion_matrix(pred_left['label'], pred_left['predict'], classes=class_names)
plt.savefig('provaconfusion_left.png', dpi=300)


# In[50]:


metrics_left = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(pred_left['label'], pred_left['predict'])],
    'recall':[recall_score(pred_left['label'], pred_left['predict'], average='macro')],
    'precision':[precision_score(pred_left['label'], pred_left['predict'], average='macro')]
    },
orient = 'index')


# In[51]:


metrics_left


# In[52]:


pred_total = (df_MLOL.iloc[:,0:4] + df_CCL.iloc[:,0:4] + df_MLOR.iloc[:,0:4] + df_CCR.iloc[:,0:4])/4


# In[53]:


pred_total


# In[54]:


pred_total['label'] = label


# In[55]:


prediction_TOT = np.argmax((pred_total.iloc[:,0:4]).to_numpy(), axis=1)


# In[56]:


pred_total['predict'] = prediction_TOT
pred_total.head()


# In[57]:


plot_confusion_matrix(pred_total['label'], pred_total['predict'], classes=class_names)
plt.savefig('provaconfusion_total.png', dpi=300)


# In[58]:


metrics_total = pd.DataFrame.from_dict(
    {
    'accuracy':[accuracy_score(pred_total['label'], pred_total['predict'])],
    'recall':[recall_score(pred_total['label'], pred_total['predict'], average='macro')],
    'precision':[precision_score(pred_total['label'], pred_total['predict'], average='macro')]
    },
orient = 'index')


# In[59]:


metrics_total


# In[61]:


f_train = open('1/acc_train.txt', 'r')
f_val = open('1/acc_val.txt', 'r')

datt=[]
datv=[]

for ln in f_train:
    datt.append(ln)
    
for ln in f_val:
    datv.append(ln)   
    
f_train.close()
f_val.close()

x=np.arange(1,101)

datt=np.array(datt)
datt=datt.astype(np.float)

datv=np.array(datv)
datv=datv.astype(np.float)


plt.plot(x, datt, 'r-', label='Training accuracy')
plt.plot(x, datv, 'b-', label='Validation accuracy')
plt.title("Accuracy over epochs for 450x450 pixels images")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy.png')
#plt.show()


# In[59]:


f_train = open('1/loss_train.txt', 'r')
f_val = open('1/loss_val.txt', 'r')

datt=[]
datv=[]

for ln in f_train:
    datt.append(ln)
    
for ln in f_val:
    datv.append(ln)   
    
f_train.close()
f_val.close()

x=np.arange(1,101)

datt=np.array(datt)
datt=datt.astype(np.float)

datv=np.array(datv)
datv=datv.astype(np.float)


plt.plot(x, datt, 'r-', label='Training loss function')
plt.plot(x, datv, 'b-', label='Validation loss function')
plt.title("Loss function over epochs for 450x450 pixels images")
plt.xlabel("Epochs")
plt.ylabel("Loss function")
plt.legend()
plt.savefig('Loss_function.png')
#plt.show()


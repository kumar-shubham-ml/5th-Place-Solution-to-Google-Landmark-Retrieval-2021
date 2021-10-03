#!/usr/bin/env python
# coding: utf-8

# In[30]:

import os
# In[31]:


"""
To update Tensorflow Version on TPU (If needed)
"""

# !pip install -U tensorflow==2.4.0

# print("update TPU server tensorflow version...")

# !pip install cloud-tpu-client
import tensorflow as tf 
# from cloud_tpu_client import Client
# print(tf.__version__)
# Client().configure_tpu_version(tf.__version__, restart_type='ifNeeded')


# In[32]:


# dataset_name = 'glr-arcface-256-b0-61242'
# !kaggle datasets download ks2019/{dataset_name} -q -p kaggle_datasets/zipfiles/


# In[33]:


# !unzip kaggle_datasets/zipfiles/{dataset_name}


# In[34]:


# In[35]:

model_name='v2l-12epochs-720T720'
model_path = "gs://ks-bkt-eu/models-new/v2l-12epochs-800/EF6_fold1_last.h5"


# In[36]:
TPU_NAME = 'ks-tpu'

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()
    
AUTO = tf.data.experimental.AUTOTUNE

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[37]:


import re
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import backend as K
# import tensorflow_addons as tfa
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
import tensorflow_hub as tfhub


# In[38]:


# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# In[39]:


EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

def freeze_BN(model):
    # Unfreeze layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

# Function to create our EfficientNetB3 model
def get_model():

    if model_head=='arcface':
        head = ArcMarginProduct
    elif model_head=='curricularface':
        head = CurricularFace
    else:
        assert 1==2, "INVALID HEAD"
    
    with strategy.scope():

        margin = head(
            n_classes = N_CLASSES, 
            s = 30, 
            m = 0.6, 
            name='head/arc_margin', 
            dtype='float32'
            )

        inp = tf.keras.layers.Input(shape = [IMAGE_SIZE, IMAGE_SIZE, 3], name = 'inp1')
        label = tf.keras.layers.Input(shape = (), name = 'inp2')
        
        if model_type == 'effnetv1':
            x = EFNS[EFF_NET](weights = 'noisy-student', include_top = False)(inp)
            embed = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif model_type == 'effnetv2':
            FEATURE_VECTOR = f'{EFFNETV2_ROOT}/tfhub_models/efficientnetv2-{EFF_NETV2}/feature_vector'
            embed = tfhub.KerasLayer(FEATURE_VECTOR, trainable=True)(inp)
            
        embed = tf.keras.layers.Dropout(0.2)(embed)
        embed = tf.keras.layers.Dense(512)(embed)
        x = margin([embed, label])
        
        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
        embed_model = tf.keras.models.Model(inputs = inp, outputs = embed)  
        
        return model,embed_model


# In[41]:


N_CLASSES = 203094
IMAGE_SIZE = 720
model_head = 'arcface'
model_type = 'effnetv2'
EFF_NET = 0
EFF_NETV2 = 'l-21k-ft1k'
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
TTA = ['rotate0',  'rotate0_lr','rotate0_ud_lr','rotate0_ud']
TTA = ['rotate0',  'rotate0_lr']


# In[42]:


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
#         "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    return example['image']

def read_names(example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
#         "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    return example['image_name']

def load_names(filenames, ordered = False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
#     dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_names, num_parallel_calls = AUTO) 
    return dataset

# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered = False):
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
#     dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = AUTO) 
    return dataset


# Function to decode our images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, [IMAGE_SIZE,IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Function to read our test image and return image
def read_image(image):
    image = tf.io.read_file(image)
    image = decode_image(image)
    return image

def test_time_augmentation(img,tta=None):
    if tta:
        if tta[-3:]=='_lr':
            img = tf.image.flip_left_right(img)
            tta = tta[:-3]

        if tta[-3:]=='_ud':
            img = tf.image.flip_up_down(img)
            tta = tta[:-3]
    return img
    
# Function to get our dataset that read images
def get_test_dataset(filenames,tta=None):
    dataset = load_dataset(filenames, ordered = True)
    dataset = dataset.map(lambda image: decode_image(image), num_parallel_calls = AUTO)
    dataset = dataset.map(lambda image: test_time_augmentation(image,tta), num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_names(filenames,tta=None):
    dataset = load_names(filenames, ordered = True)
    return dataset

# In[44]:


EFFNETV2_ROOT ='gs://ks-bkt-eu/ks-bkt/efficientnet-v2-tfhub'

FILENAMES = [f'gs://ks-bkt-new/train/landmark-2021-gld2-{x}.tfrec' for x in range(20)]


# In[47]:


INDEX_FILENAMES = [f'gs://ks-bkt-new/tfr/val-tfrecords/landmark-2021-val-index-{x}-253919.tfrec' for x in range(1,4)]


# In[48]:


TEST_FILENAMES = ['gs://ks-bkt-new/tfr/val-tfrecords/landmark-2021-val-test-117577.tfrec']


# In[49]:


model,embed_model = get_model()


# In[50]:


os.system(f'gsutil cp {model_path} model.h5')


# In[51]:


model.load_weights("model.h5")

#embed_model.save('model.h5')
# In[52]:


model.summary()


# In[53]:




# In[54]:


from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io


# In[55]:


gcs_model_path = f'gs://ks-bkt-eu/predictions/{model_name}/model.h5'
# Save Keras ModelCheckpoints locally
# embed_model.save('model1.h5')

os.system(f'gsutil -m cp model.h5 {gcs_model_path}')


# In[56]:


from tqdm.auto import tqdm
for fold,filename in enumerate(TEST_FILENAMES):
    save_path = f'gs://ks-bkt-eu/predictions/{model_name}/test-predictions-{fold}.npy'
    names_path = f'gs://ks-bkt-eu/predictions/{model_name}/test-names-{fold}.npy'
    dataset = get_test_dataset([filename])
    predictions = embed_model.predict(dataset,verbose=1)
    names = get_test_names([filename])
    names = next(iter(names.batch(predictions.shape[0]))).numpy()
    n_samples = predictions.shape[0]
    np.save(file_io.FileIO(save_path, 'w'), predictions)
    np.save(file_io.FileIO(names_path, 'w'), names)


# In[ ]:


from tqdm.auto import tqdm
for fold,filename in enumerate(TEST_FILENAMES):
    save_path = f'gs://ks-bkt-eu/predictions/{model_name}/test-predictions-lr-{fold}.npy'
    names_path = f'gs://ks-bkt-eu/predictions/{model_name}/test-names-lr-{fold}.npy'
    dataset = get_test_dataset([filename],tta='flip_lr')
    predictions = embed_model.predict(dataset,verbose=1)
    names = get_test_names([filename])
    names = next(iter(names.batch(predictions.shape[0]))).numpy()
    n_samples = predictions.shape[0]
    np.save(file_io.FileIO(save_path, 'w'), predictions)
    np.save(file_io.FileIO(names_path, 'w'), names)


# In[ ]:


from tqdm.auto import tqdm
for fold,filename in enumerate(INDEX_FILENAMES):
    save_path = f'gs://ks-bkt-eu/predictions/{model_name}/index-predictions-{fold}.npy'
    names_path = f'gs://ks-bkt-eu/predictions/{model_name}/index-names-{fold}.npy'
    dataset = get_test_dataset([filename])
    predictions = embed_model.predict(dataset,verbose=1)
    names = get_test_names([filename])
    names = next(iter(names.batch(predictions.shape[0]))).numpy()
    n_samples = predictions.shape[0]
    np.save(file_io.FileIO(save_path, 'w'), predictions)
    np.save(file_io.FileIO(names_path, 'w'), names)


# In[ ]:


from tqdm.auto import tqdm
for fold,filename in enumerate(INDEX_FILENAMES):
    save_path = f'gs://ks-bkt-eu/predictions/{model_name}/index-predictions-lr-{fold}.npy'
    names_path = f'gs://ks-bkt-eu/predictions/{model_name}/index-names-lr-{fold}.npy'
    dataset = get_test_dataset([filename],tta='flip_lr')
    predictions = embed_model.predict(dataset,verbose=1)
    names = get_test_names([filename])
    names = next(iter(names.batch(predictions.shape[0]))).numpy()
    n_samples = predictions.shape[0]
    np.save(file_io.FileIO(save_path, 'w'), predictions)
    np.save(file_io.FileIO(names_path, 'w'), names)


# In[ ]:


from tqdm.auto import tqdm
for fold,filename in enumerate(FILENAMES):
        save_path = f'gs://ks-bkt-eu/predictions/{model_name}/train-predictions-{fold}.npy'
        names_path = f'gs://ks-bkt-eu/predictions/{model_name}/train-names-{fold}.npy'
        dataset = get_test_dataset([filename])
        predictions = embed_model.predict(dataset,verbose=1)
        names = get_test_names([filename])
        names = next(iter(names.batch(predictions.shape[0]))).numpy()
        n_samples = predictions.shape[0]
        np.save(file_io.FileIO(save_path, 'w'), predictions)
        np.save(file_io.FileIO(names_path, 'w'), names)


# In[ ]:


for fold,filename in enumerate(FILENAMES):
    save_path = f'gs://ks-bkt-eu/predictions/{model_name}/train-predictions-lr-{fold}.npy'
    names_path = f'gs://ks-bkt-eu/predictions/{model_name}/train-names-lr-{fold}.npy'
    dataset = get_test_dataset([filename],tta='flip_lr')
    predictions = embed_model.predict(dataset,verbose=1)
    names = get_test_names([filename])
    names = next(iter(names.batch(predictions.shape[0]))).numpy()
    n_samples = predictions.shape[0]
    np.save(file_io.FileIO(save_path, 'w'), predictions)
    np.save(file_io.FileIO(names_path, 'w'), names)


os.system(f'gcloud compute tpus stop {TPU_NAME} --zone=europe-west4-a')

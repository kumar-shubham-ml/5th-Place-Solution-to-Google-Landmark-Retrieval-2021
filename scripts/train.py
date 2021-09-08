#!/usr/bin/env python
# coding: utf-8


import os
import tensorflow as tf

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver('ks-tpu')
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
import tensorflow_addons as tfa
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pickle
import json
import tensorflow_hub as tfhub
from datetime import datetime
EXPERIMENT = 0

save_dir = '.'
run_ts = datetime.now().strftime('%Y%m%d-%H%M%S')
save_dir = f'/home/kumarshubham/runs/experiments-{EXPERIMENT}/{run_ts}'
print("Model ID:",run_ts)
os.makedirs(save_dir,exist_ok=True)


class config:
    
    
    SEED = 42
    FOLD_TO_RUN = 0
    FOLDS = 5
    DEBUG = False
    EVALUATE = True
    RESUME = False
    RESUME_EPOCH = 39
    # model_path = f'/content/drive/MyDrive/Kaggle/GLR-2021/experiments-{EXPERIMENT}/20210831-200946/'
    model_path = f'../input/glr-eff-v2-m-arcface-retraining-at-640/'
    
    ### Dataset
    dataset = 'v2'  # one of 'v2', 'v2c', 'comp'
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync
    IMAGE_SIZE = 384
    
    ### Model
    model_type = 'effnetv1'  # One of effnetv1, effnetv2
    EFF_NET = 7
    EFF_NETV2 = 'm-21k-ft1k'
    FREEZE_BATCH_NORM = False
    head = 'arcface' # one of arcface, curricular-face
    EPOCHS = 30
    LR = 0.001
    message='retraining 640 epoch 2'
    
    ### Augmentations
    PRECROP_IMAGE_SIZE = 512
    ROT_ = 10.0
    SHR_ = 2.0
    HZOOM_ = 8.0
    WZOOM_ = 8.0
    HSHIFT_ = 8.0
    WSHIFT_ = 8.0
    CUTOUT = False
    save_dir = save_dir

    EFFNETV2_ROOT = 'gs://ks-utils/efficientnet-v2-tfhub/'
    SNAPSHOT_THRESOLD = 99


if config.dataset=='comp':
    config.N_CLASSES = 81313
elif config.dataset=='v2':
    config.N_CLASSES = 203094


# In[ ]:


with open(config.save_dir+'/config.json', 'w') as fp:
    json.dump({x:dict(config.__dict__)[x] for x in dict(config.__dict__) if not x.startswith('_')}, fp)


# ### Dataset

# In[ ]:


if config.dataset=='v2':
    TRAINING_FILENAMES = [f'gs://landmark-2021/train/landmark-2021-gld2-{x}.tfrec' for x in range(20) if x!=config.FOLD_TO_RUN]
    VALIDATION_FILENAMES = [f'gs://landmark-2021/train/landmark-2021-gld2-{config.FOLD_TO_RUN}.tfrec']
elif config.dataset=='comp':
    GCS_PATHS = {
        0: 'gs://kds-665f16a629f52bf486a51f98309aae3de7a9faee4de89b8e97f4ae18',
        1: 'gs://kds-52a1fc8b8869ddc2cb45cad089adb9b57317df88f8dceebdc5992689',
        2: 'gs://kds-ef6eb448bbf5e9c0240f0f577574cef3f3a59575a6d65b414b78ed50',
        3: 'gs://kds-55d7fb28e710a0aaf21e71c116872a6b5980e7834ecfe39a818ca0fc',
        4: 'gs://kds-2b6f9cf229d43964e64b5e084b303bec9f07cfdbe643ddb769c0a65b'
        }

    TRAINING_FILENAMES = []
    VALIDATION_FILENAMES = []

    for fold in range(5):
        GCS_PATH = GCS_PATHS[fold]
        if fold==config.FOLD_TO_RUN:
            VALIDATION_FILENAMES += tf.io.gfile.glob(GCS_PATH + '/*.tfrec')
        TRAINING_FILENAMES += tf.io.gfile.glob(GCS_PATH + '/*.tfrec')
            
if config.DEBUG:
    TRAINING_FILENAMES = [TRAINING_FILENAMES[0]]
    VALIDATION_FILENAMES = [VALIDATION_FILENAMES[0]]
    
print(len(TRAINING_FILENAMES))
print(len(VALIDATION_FILENAMES))


# In[ ]:


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, DIM=256):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM%2 #fix for size 331
    
    rot = config.ROT_ * tf.random.normal([1], dtype='float32')
    shr = config.SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / config.HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / config.WZOOM_
    h_shift = config.HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = config.WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])


# In[ ]:


# Function to get our f1 score
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches

# Data augmentation function
def data_augment(posting_id, image, label_group, matches):

    ### CUTOUT
    if tf.random.uniform([])>0.5 and config.CUTOUT:
      N_CUTOUT = 6
      for cutouts in range(N_CUTOUT):
        if tf.random.uniform([])>0.5:
           DIM = config.IMAGE_SIZE
           CUTOUT_LENGTH = DIM//8
           x1 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           x2 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           filter_ = tf.concat([tf.zeros((x1,CUTOUT_LENGTH)),tf.ones((CUTOUT_LENGTH,CUTOUT_LENGTH)),tf.zeros((DIM-x1-CUTOUT_LENGTH,CUTOUT_LENGTH))],axis=0)
           filter_ = tf.concat([tf.zeros((DIM,x2)),filter_,tf.zeros((DIM,DIM-x2-CUTOUT_LENGTH))],axis=1)
           cutout = tf.reshape(1-filter_,(DIM,DIM,1))
           image = cutout*image

    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return posting_id, image, label_group, matches

# Function to decode our images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, [config.IMAGE_SIZE,config.IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# This function parse our images and also get the target variable
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
#         "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['image_name']
    image = decode_image(example['image'])
#     label_group = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth = N_CLASSES)
    label_group = tf.cast(example['target'], tf.int32)
#     matches = example['matches']
    matches = 1
    return posting_id, image, label_group, matches

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

# This function is to get our training tensors
def get_training_dataset(filenames, ordered = False):
    dataset = load_dataset(filenames, ordered = ordered)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# Function to count how many photos we have in
def count_data_items(filenames):
    # Assuming 200000 Images per TFRecord
    if config.dataset=='v2':
        return 200000*len(filenames)
    else:
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
print(f'Dataset: {NUM_TRAINING_IMAGES} training images')


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


# In[ ]:


#CurricularFace class keras layer
class CurricularFace(tf.keras.layers.Layer):
    '''
    Implements Curricular Face.

    Reference:
        To be added
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(CurricularFace, self).__init__(**kwargs)

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
        super(CurricularFace, self).build(input_shape[0])

        self.kernel = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
            regularizer=None)
        
        self.t = self.add_weight(
              name='t',
              shape=(1,),
              dtype='float32',
              initializer=tf.keras.initializers.get('zeros'),
              synchronization=tf.VariableSynchronization.ON_READ,
              trainable=False,
              aggregation=tf.VariableAggregation.MEAN,
              experimental_autocast=False)
    
    def _assign_new_value(self, variable, value):
        with tf.keras.backend.name_scope('AssignNewValue') as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):  # pylint: disable=protected-access
                    return tf.compat.v1.assign(variable, value, name=scope)

    def call(self, inputs):
        embeddings, label = inputs
        label = tf.cast(label,tf.int32)
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        kernel_norm = tf.math.l2_normalize(self.kernel, axis=0)
        cos_theta = tf.matmul(embeddings, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta,-1,1) # for numerical stability
        origin_cos = tf.identity(cos_theta)
        out = tf.stack([tf.range(0,tf.shape(embeddings)[0]),label],axis=1)
        target_logit = tf.gather_nd(cos_theta,tf.transpose([tf.range(0,tf.shape(embeddings)[0]),label]))

        sin_theta = tf.math.sqrt(1.0 - tf.math.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin) 
        final_target_logit = tf.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)
        
        
        one_hot = tf.cast(
                    tf.one_hot(label, depth=self.n_classes),
                    dtype=cos_theta.dtype
                )


        mask = cos_theta > cos_theta_m[:,None]
        hard_example = tf.identity(cos_theta)
        self._assign_new_value(self.t, tf.reduce_mean(target_logit) * 0.01 + (1 - 0.01) * self.t)
        cos_hard = hard_example * (self.t + hard_example)
        cos_theta = tf.where(mask, cos_hard, cos_theta)  
        cos_theta = one_hot*final_target_logit[:,None]+((1.0 - one_hot) * cos_theta)
        output = cos_theta * self.s
        return output


# In[ ]:


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

    if config.head=='arcface':
        head = ArcMarginProduct
    elif config.head=='curricularface':
        head = CurricularFace
    else:
        assert 1==2, "INVALID HEAD"
    
    with strategy.scope():

        margin = head(
            n_classes = config.N_CLASSES, 
            s = 30, 
            m = 0.6, 
            name='head/arc_margin', 
            dtype='float32'
            )

        inp = tf.keras.layers.Input(shape = [config.IMAGE_SIZE, config.IMAGE_SIZE, 3], name = 'inp1')
        label = tf.keras.layers.Input(shape = (), name = 'inp2')
        
        if config.model_type == 'effnetv1':
            x = EFNS[config.EFF_NET](weights = 'noisy-student', include_top = False)(inp)
            embed = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif config.model_type == 'effnetv2':
            FEATURE_VECTOR = f'{EFFNETV2_ROOT}/tfhub_models/efficientnetv2-{config.EFF_NETV2}/feature_vector'
            embed = tfhub.KerasLayer(FEATURE_VECTOR, trainable=True)(inp)
            
        embed = tf.keras.layers.Dropout(0.2)(embed)
        embed = tf.keras.layers.Dense(512)(embed)
        x = margin([embed, label])
        
        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
        embed_model = tf.keras.models.Model(inputs = inp, outputs = embed)  
        
        opt = tf.keras.optimizers.Adam(learning_rate = config.LR)
        if config.FREEZE_BATCH_NORM:
            freeze_BN(model)

        model.compile(
            optimizer = opt,
            loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
            ) 
        
        return model,embed_model


# In[ ]:


def get_lr_callback(plot=False):
    lr_start   = 0.000001
    lr_max     = 0.000005 * config.BATCH_SIZE  
    lr_min     = 0.000001
    lr_ramp_ep = 4
    lr_sus_ep  = 0
    lr_decay   = 0.9
   
    def lrfn(epoch):
        if config.RESUME:
            epoch = epoch + config.RESUME_EPOCH
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
        
    if plot:
        epochs = list(range(config.EPOCHS))
        learning_rates = [lrfn(x) for x in epochs]
        plt.scatter(epochs,learning_rates)
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback



class Snapshot(tf.keras.callbacks.Callback):
    
    def __init__(self,snapshot_min_epoch,fold):
        super(Snapshot, self).__init__()
        self.snapshot_min_epoch = snapshot_min_epoch
        self.fold = fold
        
        
    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
#         print(f"epoch: {epoch}, train_acc: {logs['acc']}, valid_acc: {logs['val_acc']}")
        if epoch >=self.snapshot_min_epoch: # your custom condition         
            self.model.save_weights(config.save_dir+f"/EF{config.EFF_NET}_fold{self.fold}_epoch{epoch}.h5")
        self.model.save_weights(config.save_dir+f"/EF{config.EFF_NET}_fold{self.fold}_last.h5")


# ### Training

# In[ ]:


seed_everything(config.SEED)
VERBOSE = 1
train_dataset = get_training_dataset(TRAINING_FILENAMES, ordered = False)
STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES) // config.BATCH_SIZE
if config.dataset=='v2':
  STEPS_PER_EPOCH = STEPS_PER_EPOCH//3
train_logger = tf.keras.callbacks.CSVLogger(config.save_dir+'/training-log-fold-%i.h5.csv'%config.FOLD_TO_RUN)
# BUILD MODEL
K.clear_session()
model,embed_model = get_model()
snap = Snapshot(snapshot_min_epoch=config.SNAPSHOT_THRESHOLD,fold=config.FOLD_TO_RUN)
model.summary()


# In[ ]:


if config.RESUME:   
  model.load_weights(config.model_path+f"EF{config.EFF_NET}_fold{config.FOLD_TO_RUN}_last.h5")


# In[ ]:


print('#### Image Size %i with EfficientNet B%i and batch_size %i'%
      (config.IMAGE_SIZE,config.EFF_NET,config.BATCH_SIZE))

history = model.fit(train_dataset,
                steps_per_epoch = STEPS_PER_EPOCH,
                epochs = config.EPOCHS,
                callbacks = [snap,get_lr_callback(),train_logger], 
                verbose = VERBOSE)
# model.save_weights(config.save_dir+'/fold-%i.h5')


# ### Save Optimiser 

# In[ ]:


symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(config.save_dir+'/optimizer.pkl', 'wb') as f:
    pickle.dump(weight_values, f)


# In[ ]:





import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import transforms, datasets
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import os
import h5py
from sklearn.preprocessing import normalize as l2

sys.path.append('../../python_scripts/')

# setting parameters
exp = '086'
epoch = '110'
experiment = '../../saving_model/exp' + exp + '/exp' + exp + '_epoch_' + epoch + '.pth.tar'
name_out = exp + '_L2_long_2'
batch_size = 100
# number of classes used to train
nb_classes = 16000
#16000
#16378

# define a CNN
#from networks import large_2_drop_96x96_pad_and_pool as Net
#from networks import large_2_drop_96x96_pad_and_pool as Net
#from networks import large_32x32_pad_3x3_real_pool as Net
from networks import largest as Net
net = Net(nb_classes).cuda()

# loading the model
checkpoint = torch.load(experiment)
net.load_state_dict(checkpoint['state_dict'])

# processing features

def process_layers(out_conv1, out_conv2, out_conv3, batch_size):
    # 67840
    # set the layer config for extracting the features
    out_c1 = pooling_to_fixed_value(out_conv1).view(batch_size, -1)
    out_c2 = pooling_to_fixed_value(out_conv2).view(batch_size, -1)
    out_c3 = pooling_to_fixed_value(out_conv3).view(batch_size, -1)

    out_c = torch.cat([out_c1,
                       out_c2,
                       out_c3],
                      dim = 1)
    return out_c

def process_layers_l2(out_conv1, out_conv2, out_conv3, batch_size):
    # 67840
    # set the layer config for extracting the features
    out_c1 = pooling_to_fixed_value(out_conv1).view(batch_size, -1)
    out_c2 = pooling_to_fixed_value(out_conv2).view(batch_size, -1)
    out_c3 = pooling_to_fixed_value(out_conv3).view(batch_size, -1)

    out_c1_n = torch.div(out_c1, out_c1.norm())
    out_c2_n = torch.div(out_c2, out_c2.norm())
    out_c3_n = torch.div(out_c3, out_c3.norm())

    out_c = torch.cat([out_c1_n,
                       out_c2_n,
                       out_c3_n],
                      dim = 1)
    return out_c

def process_layers_fc(out_conv1, out_conv2, out_conv3, batch_size):
    # 67840
    # set the layer config for extracting the features
    out_c1 = pooling_to_fixed_value(out_conv1).view(batch_size, -1)
    out_c2 = pooling_to_fixed_value(out_conv2).view(batch_size, -1)
    out_c3 = pooling_to_fixed_value(out_conv3).view(batch_size, -1)

    fc_feats1 = conv_list[0](out_conv3)
    fc_feats1 = F.relu(fc_feats1)
    out_fc1 = pooling_to_fixed_value(fc_feats1).view(batch_size, -1)

    fc_feats2 = conv_list[1](fc_feats1)
    fc_feats2 = F.relu(fc_feats2)
    out_fc2 = pooling_to_fixed_value(fc_feats2).view(batch_size, -1)

    out_c = torch.cat([out_c1,
                       out_c2,
                       out_c3,
                       out_fc1,
                       out_fc2],
                      dim = 1)
    return out_c

def process_layers_fc_l2(out_conv1, out_conv2, out_conv3, batch_size):
    # 67840
    # set the layer config for extracting the features
    out_c1 = pooling_to_fixed_value(out_conv1).view(batch_size, -1)
    out_c2 = pooling_to_fixed_value(out_conv2).view(batch_size, -1)
    out_c3 = pooling_to_fixed_value(out_conv3).view(batch_size, -1)

    fc_feats1 = conv_list[0](out_conv3)
    fc_feats1 = F.relu(fc_feats1)
    out_fc1 = pooling_to_fixed_value(fc_feats1).view(batch_size, -1)

    fc_feats2 = conv_list[1](fc_feats1)
    fc_feats2 = F.relu(fc_feats2)
    out_fc2 = pooling_to_fixed_value(fc_feats2).view(batch_size, -1)

    out_c1_n = torch.div(out_c1, out_c1.norm())
    out_c2_n = torch.div(out_c2, out_c2.norm())
    out_c3_n = torch.div(out_c3, out_c3.norm())
    fc1_n = torch.div(out_fc1, out_fc1.norm())
    fc2_n = torch.div(out_fc2, out_fc2.norm())

    out_c = torch.cat([out_c1_n,
                       out_c2_n,
                       out_c3_n,
                       fc1_n,
                       fc2_n],
                      dim = 1)
    return out_c


def process_features_batch(out_conv1, out_conv2, out_conv3, batch_size):
    # process the extracteed features batch by batch (batch_size, nb_feats, height, length)
    #out_c = process_layers_l2(out_conv1, out_conv2, out_conv3, batch_size)
    out_c = process_layers_fc_l2(out_conv1, out_conv2, out_conv3, batch_size)
    #out_f = out_c.squeeze().cpu().data.view(batch_size, -1)#.numpy()
    out_f = out_c.data.cpu()#.numpy()
    return out_f


# Code to transform the weights from FC to Conv
model_layers = nn.Sequential(*list(net.children()))

conv_list = []
prev_ch = 0
for layer in model_layers:
    if isinstance(layer, nn.Linear):
        fc_dict = layer.state_dict()

        # parameters for the conv layer
        filter_size = 7 # when using the padding this should be 8
        if not prev_ch == 0:    # With this the filter size is variable only in the first layer
            filter_size = 1

        channels = int(fc_dict['weight'].size()[1]/(filter_size**2))
        deep = fc_dict['weight'].size()[0]

        prev_ch = deep

        '''
        if filter_size == 8:
            conv = nn.Conv2d(channels, deep, filter_size, filter_size, padding=4)
        elif filter_size == 1:
            conv = nn.Conv2d(channels, deep, filter_size, filter_size,)
        '''

        conv = nn.Conv2d(channels, deep, filter_size, filter_size)

        conv.load_state_dict({'weight': fc_dict['weight'].view(deep,channels,filter_size,filter_size),
                              'bias': fc_dict['bias']})

        conv_list.append(conv)

for i in conv_list:
    i.cuda()


# We fix teh size of the features that we give to the SVM
feat_side = 2
pooling_to_fixed_value = nn.AdaptiveMaxPool2d(feat_side)

#mean = [0.5, 0.5, 0.5]
#std=[1, 1, 1]

# mean from dataset058:
mean = [0.3823625683879477, 0.3790166856065496, 0.3554138533338805]
std=[0.21754145353302254, 0.21271749678359336, 0.21233947166469555]

#mean = [0.383661700858527, 0.3819784115384924, 0.3588786631614881]
#std=[0.2167717755518767, 0.21201058526724945, 0.21143164036556178]

# Load the dataset
from utils import read_images_stl10 as read_images
from utils import read_labels_stl10 as read_labels

#unlab_set_x = read_images('../data/stl10_binary/unlabeled_X.bin')
test_set_x = read_images('../../data/stl10_binary/test_X.bin')
train_set_x = read_images('../../data/stl10_binary/train_X.bin')

test_set_y = read_labels('../../data/stl10_binary/test_y.bin')
train_set_y = read_labels('../../data/stl10_binary/train_y.bin')

print 'Train set information: '
print (len(train_set_x), type(train_set_x[3]))
print ''
print 'Test set information: '
print (len(test_set_x), type(test_set_x[3]))
print ''

# loading the folds:
with open('../../data/stl10_binary/fold_indices.txt', 'r') as f_folds:
    folds = f_folds.readlines()

k_folds = []
for fold_i in folds:
    k_folds.append(np.asarray(fold_i.split(' ')[:-1], dtype=int))

print "Folds loaded:"
print len(folds), type(folds), type(folds[0]), len(folds[0])
print len(k_folds[0]), type(k_folds[0][0]), k_folds[0][0]


# defining transformations:
normalize = transforms.Normalize(mean = mean, std=std)
transf = transforms.Compose([transforms.ToTensor(), normalize])

# extracting features
print 'Extracting featuers...'

st = time.time()
train_features = torch.FloatTensor()
train_set_labels = torch.LongTensor()


batch_nb = 0
for image in range(train_set_x.shape[0]/batch_size):
    inputs  = train_set_x[(batch_nb*(batch_size)):((batch_nb + 1)*(batch_size))]
    labels = train_set_y[(batch_nb*(batch_size)):((batch_nb + 1)*(batch_size))]
    batch_nb += 1

    samples = torch.FloatTensor(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])
    for n, i in enumerate(inputs):
        samples[n] = transf(i)

    samples = Variable(samples).cuda()
    labels = torch.from_numpy(labels).long()

    net.train(False)
    (out_conv1, out_conv2, out_conv3) = net.forward_all_conv_feat_after_relu(samples)

    out_f = process_features_batch(out_conv1, out_conv2, out_conv3, batch_size)

    train_features = torch.cat([train_features, out_f], dim = 0)
    train_set_labels = torch.cat([train_set_labels, labels], dim = 0)

end = time.time()

print 'Features size for training: ' + str(train_features.size())

st = time.time()
test_features = torch.FloatTensor()
test_set_labels = torch.LongTensor()

batch_nb = 0
for image in range(test_set_x.shape[0]/batch_size):
    inputs  = test_set_x[(batch_nb*(batch_size)):((batch_nb + 1)*(batch_size))]
    labels = test_set_y[(batch_nb*(batch_size)):((batch_nb + 1)*(batch_size))]
    batch_nb += 1

    samples = torch.FloatTensor(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])
    for n, i in enumerate(inputs):
        samples[n] = transf(i)

    samples = Variable(samples).cuda()
    labels = torch.from_numpy(labels).long()

    net.train(False)
    (out_conv1, out_conv2, out_conv3) = net.forward_all_conv_feat_after_relu(samples)

    out_f = process_features_batch(out_conv1, out_conv2, out_conv3, batch_size)

    test_features = torch.cat([test_features, out_f], dim = 0)
    test_set_labels = torch.cat([test_set_labels, labels], dim = 0)


print 'Features size for testing: ' + str(test_features.size())

print 'Features extracted!'
print 'Saving features...'

# save features in a hdf5 temporal file
X_train = train_features.numpy()
y_train = train_set_labels.numpy()
X_val = test_features.numpy()
y_val = test_set_labels.numpy()


f_h5py = h5py.File(name_out+'_train.hdf5', 'w')


f_h5py.create_dataset(name='data', shape=(X_train.shape[0], X_train.shape[1]), dtype=np.float64)
f_h5py.create_dataset(name='labels', shape=(y_train.shape[0],1))

for sample in range(len(X_train)):
    #print train_x[sample].shape
    f_h5py['data'][sample, ...] = X_train[sample]
    f_h5py['labels'][sample, ...] = y_train[sample]

f_h5py.close()

f_h5py = h5py.File(name_out+'_val.hdf5', 'w')


f_h5py.create_dataset(name='data', shape=(X_val.shape[0], X_val.shape[1]), dtype=np.float64)
f_h5py.create_dataset(name='labels', shape=(y_val.shape[0],1))

for sample in range(len(X_val)):
    f_h5py['data'][sample, ...] = X_val[sample]
    f_h5py['labels'][sample, ...] = y_val[sample]

f_h5py.close()
print 'Done!'

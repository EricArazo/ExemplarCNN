import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from matplotlib import pyplot as plt


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import os
import time

import numpy as np
from PIL import Image

from networks import largest as Net

####################################### CHANGE THE NAME OF THE EXPERIMENT #######################################
experiment = '083'
initial_epoch = 0
resume = False
#################################################################################################################

nb_epochs = 110

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # Deal with "the best" model...
'''
Differences:
- 128 batch_size
- padding
- diferent reduction factor for the LR
- different LR reduction policy (longer training)
- Increase the WD with the deepness in the model
- Lower LR in the last layer
- 2.0* LR for the biases always
- 0.0 WD for the biases always

Added now:
- 3x3 maxpooling

'''

# preparing datasets for training and validation
batch_size = 128
mean = [0.383661700858527, 0.3819784115384924, 0.3588786631614881]
std=[0.2167717755518767, 0.21201058526724945, 0.21143164036556178]
normalize = transforms.Normalize(mean = mean,
                                 std=std)
train_set = datasets.ImageFolder('../surrogate_dataset/unlab_dataset_035/train_set/',
                                  transform = transforms.Compose([transforms.ToTensor(), normalize]))
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           )
val_set = datasets.ImageFolder('../surrogate_dataset/unlab_dataset_035/val_set/',
                                  transform = transforms.Compose([transforms.ToTensor(), normalize]))
val_loader = torch.utils.data.DataLoader(val_set,
                                           batch_size = batch_size,
                                           shuffle = False,
                                           )

nb_classes = len(os.listdir('../surrogate_dataset/unlab_dataset_035/train_set/'))
print "Training with " + str(nb_classes) + " classes"

# define a CNN
net = Net(nb_classes).cuda()
print "Model defined"
print "Model to GPU"

initial_lr = 0.01

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = initial_lr, momentum = 0.9, weight_decay = 1e-4)
lr = optimizer.param_groups[0]['lr']
print "Initial learning rate: " + str(lr)

# Start training
loss_history = []
accuracy_val_history = []
accuracy_train_history = []
val_loss_history = []
plot_filters = False

#################### Printing optimizer configuration ########################
print ''
print "Optimizer (initial): "
print "\tDampening: ",
print optimizer.param_groups[0]['dampening']
print "\tNesterov: ",
print optimizer.param_groups[0]['nesterov']
#print "\tparams: ",
#print optimizer.param_groups[0]['params']
print "\tLR: ",
print optimizer.param_groups[0]['lr']
print "\tWeight_decay: ",
print optimizer.param_groups[0]['weight_decay']
print "\tMomentum: ",
print optimizer.param_groups[0]['momentum']
#{'dampening': 0, 'nesterov': False, 'params': 1, , 'lr': 0.01, 'weight_decay': 0, 'momentum': 0.9}
print ''
################################################################

# Changing WD and LRs
WD = 0.004
current_lr = initial_lr

#### Following initialization is for an optimizer that deals with the bias and the weights the same way
'''
optimizer = optim.SGD([{'params': net.conv1.parameters(), 'weight_decay': WD*0.0},
                       {'params': net.conv2.parameters(), 'weight_decay': WD*0.0},
                       {'params': net.conv3.parameters(), 'weight_decay': WD*0.25},
                       {'params': net.fc1.parameters(), 'weight_decay': WD*1.0},
                       {'params': net.fc2.parameters(), 'lr': current_lr*0.25 , 'weight_decay': WD*4.0}],
                      lr = 0.01, momentum = 0.9)
'''

#### Following initialization is for an optimizer that distinguishes between bias and weights
#### Is the same as the code provided in the github:
### https://github.com/yihui-he/Exemplar-CNN/blob/master/data/nets_config/64c5-128c5-256c5-512f/template/train.prototxt

optimizer = optim.SGD([{'params': net.conv1.weight, 'weight_decay': WD*0.0},
                       {'params': net.conv1.bias, 'lr': current_lr*2.0, 'weight_decay': WD*0.0},
                       {'params': net.conv2.weight, 'weight_decay': WD*0.0},
                       {'params': net.conv2.bias, 'lr': current_lr*2.0, 'weight_decay': WD*0.0},
                       {'params': net.conv3.weight, 'weight_decay': WD*0.25},
                       {'params': net.conv3.bias, 'lr': current_lr*2.0, 'weight_decay': WD*0.0},
                       {'params': net.fc1.weight, 'weight_decay': WD*1.0},
                       {'params': net.fc1.bias, 'lr': current_lr*2.0, 'weight_decay': WD*0.0},
                       {'params': net.fc2.weight, 'lr': current_lr*0.25 , 'weight_decay': WD*4.0},
                       {'params': net.fc2.bias, 'lr': current_lr*2.0, 'weight_decay': WD*0.0}],
                      lr = current_lr, momentum = 0.9)

#################### Printing optimizer configuration ########################
print ''
print "Optimizer (cahnging parameters): "
print "\tDampening: ",
print [str(i['dampening']) for i in optimizer.param_groups]
print "\tNesterov: ",
print [str(i['nesterov']) for i in optimizer.param_groups]
#print "\tparams: ",
#print optimizer.param_groups[0]['params']
print "\tLR: ",
print [str(i['lr']) for i in optimizer.param_groups]
print "\tWeight_decay: ",
print [str(i['weight_decay']) for i in optimizer.param_groups]
print "\tMomentum: ",
print [str(i['momentum']) for i in optimizer.param_groups]
#{'dampening': 0, 'nesterov': False, 'params': 1, , 'lr': 0.01, 'weight_decay': 0, 'momentum': 0.9}
print ''
################################################################
st = time.time()

################## RESUMING TRAINING #####################3
if resume:
    model_path = '../saving_model/exp' + str(experiment) + '/exp' + str(experiment) + '_epoch_' + str(initial_epoch) + '.pth.tar'
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    #################### Printing optimizer configuration ########################
    print ''
    print "Optimizer (After loading model): "
    print "\tDampening: ",
    print [str(i['dampening']) for i in optimizer.param_groups]
    print "\tNesterov: ",
    print [str(i['nesterov']) for i in optimizer.param_groups]
    #print "\tparams: ",
    #print optimizer.param_groups[0]['params']
    print "\tLR: ",
    print [str(i['lr']) for i in optimizer.param_groups]
    print "\tWeight_decay: ",
    print [str(i['weight_decay']) for i in optimizer.param_groups]
    print "\tMomentum: ",
    print [str(i['momentum']) for i in optimizer.param_groups]
    #{'dampening': 0, 'nesterov': False, 'params': 1, , 'lr': 0.01, 'weight_decay': 0, 'momentum': 0.9}
    print ''
    ################################################################
    print ('Resuming training in epoch ' + str(initial_epoch))
    accuracy_train_history = checkpoint['history_train_acc']
    loss_history = checkpoint['history_train_loss']
    val_loss_history = checkpoint['history_val_loss']
    accuracy_val_history = checkpoint['history_val_acc']


for epoch in range(initial_epoch, nb_epochs):
    print 'Training epoch ' + str(epoch + 1).zfill(3)
    st = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(),  Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # loss
        running_loss += loss.data[0]
        loss_history.append(loss.data[0])

        # accuracy
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        accuracy_train_history.append(100 * correct / float(total))
        if i % 100 == 99:    # print every 100 mini-batches
            print('Epoch %d, iteration %5d, train_loss %.3f and train_accuracy %.3f' %
                  (epoch + 1, i + 1, running_loss / 100, accuracy_train_history[-1]))#np.mean(accuracy_train_history)))
            running_loss = 0.0
    # epoch finished.

    nd = time.time()
    print('Train time: ' + str(nd-st))

    lr = [i['lr'] for i in optimizer.param_groups]
    print "Current learning rate:",
    print lr

    """
    # Dealing with the learning rate:
    """

    if epoch == 75 or epoch == 92:
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[idx]*0.4
        print 'New learning rate: ',
        lr = [i['lr'] for i in optimizer.param_groups]
        print lr

    if epoch == 85 or epoch == 100:
        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr[idx]*0.25
        print 'New learning rate: ',
        lr = [i['lr'] for i in optimizer.param_groups]
        print lr

    # evaluating on validation set
    correct = 0
    total = 0
    running_val_loss = 0
    print 'Testing on validation...'
    st = time.time()
    for data in val_loader:
        images, labels = data
        images, labels = Variable(images).cuda(),  Variable(labels).cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        loss = criterion(outputs, labels)
        running_val_loss += loss.data[0]

    nd = time.time()
    print('Validation loss: %.3f' % (running_val_loss / (len(val_set)/batch_size)))   # we divide the loss by
                                                                                                    # the number of itereations
                                                                                                    # needed to see all the validation set
    print('Validation accuracy: %.3f %%' % (100 * correct / float(total)))
    print('Validation time: ' + str(nd-st))
    val_loss_history.append(running_val_loss /  (len(val_set)/batch_size))
    accuracy_val_history.append(100 * correct / total)

    # save the model and the statistics
    path = '../saving_model/exp' + str(experiment) + '/exp' + str(experiment) + '_epoch_' + str(epoch+1) + '.pth.tar'
    if not os.path.exists('../saving_model/exp' + str(experiment)):
        os.mkdir('../saving_model/exp' + str(experiment) )
    is_best = False
    print('Saving model...')
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'history_train_loss': loss_history,
                'history_train_acc': accuracy_train_history,
                'history_val_loss': val_loss_history,
                'history_val_acc': accuracy_val_history
            }, filename = path)

    print('Train accuracy: %.3f %%' % (accuracy_train_history[-1]))#np.mean(accuracy_train_history)))


end = time.time()
print('Finished Training in ' + str(end-st) + ' seconds.')
print('Saving trained model...')
path = '../saving_model/exp' + str(experiment) + '.pth.tar'

save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'history_train_loss': loss_history,
            'history_train_acc': accuracy_train_history,
            'history_val_loss': val_loss_history,
            'history_val_acc': accuracy_val_history
        }, filename = path)

print ('Model saved in: ' + path)

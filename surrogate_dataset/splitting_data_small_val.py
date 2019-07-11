import os
from shutil import copyfile
import numpy as np
from tqdm import tqdm
np.random.seed(42)
# create folder for train set
# create folder for validation set
try:
    os.mkdir('train_set')
    os.mkdir('val_set')
except:
    print 'Split already done!'

val_size = 0.1                # percentage/100 of the full dataset for validation
train_size = 1 - val_size     # only valid if there is no test set

classes = os.listdir('./data/') # create list with all the class names (folders)
classes.sort()

for class_i in tqdm(classes):
    try:
        os.mkdir('./train_set/' + str(class_i).zfill(len(str(len(classes)))))
        #print 'ok'
        os.mkdir('./val_set/' + str(class_i).zfill(len(str(len(classes)))))
    except:
        print 'Class folder already existing... ???'
    images = os.listdir('./data/' + class_i)
    images.sort()
    for image in images:
        if np.random.uniform(0,1) >= train_size:
            copyfile('./data/' + str(class_i).zfill(len(str(len(classes)))) + '/' + image,
                     './val_set/' + str(class_i).zfill(4) + '/' + image)
        else:
            copyfile('./data/' + str(class_i).zfill(len(str(len(classes)))) + '/' + image,
                     './train_set/' + str(class_i).zfill(len(str(len(classes)))) + '/' + image)
        # change the previous else for an elif in
        # the case that you want a test split as well...

    #print 'Class ' + str(class_i).zfill(4) + ' done!'

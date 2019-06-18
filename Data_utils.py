'''
this file *Data_utils* is responsible for loading  and preprocessing dataset (resizing & normalizing).
'''
from keras.utils import to_categorical
import glob
import pathlib
import cv2
import numpy as np
import os

'''
these variable are descriping the dataset

'''
num_of_classes = 10  # number of classes
num_of_training_example = 640  # number of traaining examples (640 video)
num_of_testing_example = 140   # number of testing example (140 video)
num_of_validation = 140     # number of valdiation examples (140 video)
IMAGE_WIDTH = 224       # frame width
IMAGE_HEIGHT = 224      # frame height
IMAGE_CHANNELS = 3      # number of channel (RGB frames)
# the path of  the pretrained model (kinetics model)
kinetics_checkpoint = 'rgb_scratch_kin600/model.ckpt'
model_checkpoint = 'my_model/model.ckpt'

# mapping each class with a unique integer
label_mapping = {'Diving-Side': 0,
                 'Golf': 1,
                 'Kicking': 2,
                 'Lifting': 3,
                 'Riding-Horse': 4,
                 'Run-Side': 5,
                 'SkateBoarding-Front': 6,
                 'Swing-Bench': 7,
                 'Swing-SideAngle': 8,
                 'Walk-Front': 9}

class_names = {0: 'Diving-Side',
                 1: 'Golf',
                 2: 'Kicking',
                 3: 'Lifting',
                 4: 'Riding-Horse',
                 5: 'Run-Side',
                 6: 'SkateBoarding-Front',
                 7: 'Swing-Bench',
                 8: 'Swing-SideAngle',
                 9: 'Walk-Front'
}

'''
this function converts path of the video to the video iteself (frames)
and normalizing pixels values to range [-1:1]
and resizing each frame.
'''
def read_and_preprocess_video(video_folder, num_frames):

    images = np.zeros((1, num_frames, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.float32)
    # print(images.shape)
    i = 0
    for file_name in sorted(glob.glob(video_folder)):

        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image / 255.0
        image = image * 2
        image = image - 1
        images[0][i] = image
        i = i + 1

    return images

'''
This function will load the training dataset it will load:
-all video paths (str)
-all video labels (int)
-all video frames number (int)
and return all 3 lists.
'''
def load_training_data():
    data_root = os.getcwd()
    data_root += '/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/train/*'))

    all_video_paths = np.array(all_video_paths, dtype=np.str)
    all_video_paths = [x + '/*.jpg' for x in all_video_paths] # read only jpg files from video folder.
    video_labels = np.zeros((num_of_training_example, 1))

    num_video_frames = np.zeros(shape=(num_of_training_example), dtype=np.int32)

    for i in range(0, num_of_training_example):
        class_name = all_video_paths[i].split('/')[-4]
        video_labels[i] = label_mapping[class_name]
        k = 0
        for _ in glob.glob(all_video_paths[i]):
            k += 1
        num_video_frames[i] = k
    '''
    shuffling the dataset.
    '''
    np.random.seed(1)
    p = np.random.permutation(num_of_training_example)
    video_labels = video_labels[p]
    num_video_frames = num_video_frames[p]
    all_video_paths = [all_video_paths[k] for k in p]
    '''
    convert the label to one-hot vector.
    ex label ->1   to   [1 0 0 0 0 0 0 0 0 0]
    '''
    video_labels = to_categorical(video_labels)

    return all_video_paths, video_labels, num_video_frames

'''
This function will load the testing dataset it will load:
-all video paths (str)
-all video labels (int)
-all video frames number (int)
and return all 3 lists.
'''
def load_testing_data():
    data_root = os.getcwd()
    data_root += '/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/test/*'))
    all_video_paths = np.array(all_video_paths, dtype=np.str)
    all_video_paths = [x + '/*.jpg' for x in all_video_paths]  # read only jpg files from video folder.

    video_labels = np.zeros((num_of_testing_example))
    num_video_frames = np.zeros(shape=(num_of_validation), dtype=np.int32)

    for i in range(0, num_of_testing_example):
        class_name = all_video_paths[i].split('/')[-4]
        video_labels[i] = label_mapping[class_name]
        k = 0
        for _ in glob.glob(all_video_paths[i]):
            k += 1
        num_video_frames[i] = k
    '''
    shuffle the test dataset
    '''
    np.random.seed(2)
    p = np.random.permutation(num_of_testing_example)
    video_labels = video_labels[p]
    num_video_frames = num_video_frames[p]
    all_video_paths = [all_video_paths[i] for i in p]
    '''
        convert the label to one-hot vector.
        ex label ->1   to   [1 0 0 0 0 0 0 0 0 0]
        '''
    video_labels = to_categorical(video_labels)
    return all_video_paths, video_labels, num_video_frames

'''
This function will load the validation dataset it will load:
-all video paths (str)
-all video labels (int)
-all video frames number (int)
and return all 3 lists.
'''

def load_validation_data():
    data_root = os.getcwd()
    data_root += '/ucf_action'
    data_root = pathlib.Path(data_root)
    all_video_paths = list(data_root.glob('*/validation/*'))
    all_video_paths = np.array(all_video_paths, dtype=np.str)
    all_video_paths = [x + '/*.jpg' for x in all_video_paths] # read only jpg files from video folder.
    video_labels = np.zeros((num_of_validation))

    num_video_frames = np.zeros(shape=(num_of_validation), dtype=np.int32)

    for i in range(0, num_of_validation):
        class_name = all_video_paths[i].split('/')[-4]
        video_labels[i] = label_mapping[class_name]
        k = 0
        for _ in glob.glob(all_video_paths[i]):
            k += 1
        num_video_frames[i] = k

    '''
    shuffling the dataset
    '''
    np.random.seed(2)
    p = np.random.permutation(num_of_validation)
    video_labels = video_labels[p]
    num_video_frames = num_video_frames[p]
    all_video_paths = [all_video_paths[i] for i in p]
    '''
        convert the label to one-hot vector.
        ex label ->1   to   [1 0 0 0 0 0 0 0 0 0]
    '''
    video_labels = to_categorical(video_labels)
    return all_video_paths, video_labels, num_video_frames
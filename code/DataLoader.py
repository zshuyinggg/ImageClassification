import os
import pickle
import numpy as np
""" 
This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    def unpickle(filepath):
        with open(filepath, 'rb') as fo:
            df = pickle.load(fo, encoding = 'latin-1')
        return df

    def get_data(pth:str):
        # Unpack data to float32 for data; int32 for labels
        data = unpickle(pth)
        return data['data'].astype(np.float32), np.array(data['labels']).astype(np.int32)

    def shuffle_data(arr1, arr2, seed=0):
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(arr1))
        return arr1[p], arr2[p]

    def load_datafiles(pth_list):
        data, labels = get_data(pth_list[0])
        for pth in pth_list[1:]:
            data_temp, labels_temp = get_data(pth)
            data = np.concatenate((data, data_temp), axis=0)
            labels = np.append(labels, labels_temp)
        return data, labels

    fn_train = (
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5',
    )

    fn_test = 'test_batch'

    pths_train = [os.path.join(data_dir, fn) for fn in fn_train]
    pth_test = os.path.join(data_dir, fn_test)

    x_train, y_train = load_datafiles(pths_train)
    x_test, y_test = get_data(pth_test)

    # Shuffle
    x_train, y_train = shuffle_data(x_train, y_train, seed=1211)
    x_test, y_test = shuffle_data(x_test, y_test, seed=1123)

    # Check data shapes and types
    assert x_train.shape == (50000, 3072)
    assert y_train.shape == (50000,)
    assert x_test.shape == (10000, 3072)
    assert y_test.shape == (10000,)
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert y_train.dtype == np.int32
    assert y_test.dtype == np.int32
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

# def load_data(data_dir):
#     """ Load the CIFAR-10 dataset.

#     Args:
#         data_dir: A string. The directory where data batches are stored.
    
#     Returns:
#         x_train: An numpy array of shape [50000, 3072]. 
#         (dtype=np.float32)
#         y_train: An numpy array of shape [50000,]. 
#         (dtype=np.int32)
#         x_test: An numpy array of shape [10000, 3072]. 
#         (dtype=np.float32)
#         y_test: An numpy array of shape [10000,]. 
#         (dtype=np.int32)
#     """
#     ### YOUR CODE HERE
#     data_list=[]
#     label_list=[]
#     for f_name in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
#         f_path=os.path.join(data_dir,f_name)
#         dic=unpickle(f_path)
#         data_list.append(dic[b'data'])
#         label_list+=(dic[b'labels']
#         )
#     test_path=os.path.join(data_dir,'test_batch')
#     test_data,test_label=unpickle(test_path)[b'data'],unpickle(test_path)[b'labels']
#     data_all=np.vstack(data_list)


#     x_train,y_train=data_all,np.array(label_list)
#     x_test,y_test=test_data,np.array(test_label)
#     # print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#     ### YOUR CODE HERE

#     return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """
    

    ### YOUR CODE HERE
    images=np.load(data_dir)
    ### END CODE HERE

    return images

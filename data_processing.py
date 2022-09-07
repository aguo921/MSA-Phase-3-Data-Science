import numpy as np
import pickle

# extract dict object from file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# reshape image from (3072,) to (32,32,3)
def reshape_image(data):
    data = np.reshape(data, (3,32,32))
    data = np.moveaxis(data, 0, -1)

    return data

# load training and test batches
def load_batches():
    train_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
    train_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
    train_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
    train_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
    train_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
    test_batch = unpickle('cifar-10-batches-py/test_batch')

    return (train_batch_1, train_batch_2, train_batch_3, train_batch_4, train_batch_5), test_batch

# combine batches into single batch
def combine_batches(batches):
    combined_batch = {
        b'data': [],
        b'labels': []
    }

    for batch in batches:
        for i in range(len(batch[b'data'])):
            combined_batch[b'data'].append(batch[b'data'][i])
            combined_batch[b'labels'].append(batch[b'labels'][i])

    return combined_batch

# return x and y vectors from batches
def process_batch(batch, label, n):
    x = []
    y = []

    true_count = 0
    false_count = 0

    for i in range(len(batch[b'data'])):
        if batch[b'labels'][i] == label:
            if true_count < n:
                x.append(reshape_image(batch[b'data'][i]))
                y.append(1)
                true_count += 1
        else:
            if false_count < n:
                x.append(reshape_image(batch[b'data'][i]))
                y.append(0)
                false_count += 1
    
    return np.array(x), np.array(y)

# display image
def show_image(axs, data):
    axs.imshow(data)
    axs.axis('off')
import pandas as pd
import numpy as np

# Assume tar.gz file to be extracted at project directory.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape_image(data):
    data = np.reshape(data, (3,32,32))
    data = np.moveaxis(data, 0, -1)

    return data

def load_batches():
    train_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
    train_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
    train_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
    train_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
    train_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
    test_batch = unpickle('cifar-10-batches-py/test_batch')

    return (train_batch_1, train_batch_2, train_batch_3, train_batch_4, train_batch_5), test_batch

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

# def filter_classes(batch, label):
#     x = []
#     y = []

#     true_count = 0
#     false_count = 0

#     # convert label into 1 if label matches chosen label, 0 otherwise
#     labels = np.array([int(i == label) for i in batch[b'labels']])

#     # find the 
#     _, counts = np.unique(labels, return_counts=True)
#     n = min(counts)

#     for i in range(len(batch[b'data'])):
#         if batch[b'labels'][i] == label:
#             if true_count < n:
#                 x.append(reshape_image(batch[b'data'][i]))
#                 y.append(1)
#                 true_count += 1
#         else:
#             if false_count < n:
#                 x.append(reshape_image(batch[b'data'][i]))
#                 y.append(0)
#                 false_count += 1

#     return np.array(x), np.array(y)

def show_image(axs, data):
    axs.imshow(data)
    axs.axis('off')
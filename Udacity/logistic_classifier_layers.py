import cPickle as pickle
import numpy as np

import tensorflow as tf

image_size = 28
num_class = 10
num_step = 100


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_class) == labels[:, None]).astype(np.float32)
    return dataset, labels


if __name__ == '__main__':
    # get data
    pickle_file = 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset_t = save['train_dataset']
        train_labels_t = save['train_labels']
        valid_dataset_t = save['valid_dataset']
        valid_labels_t = save['valid_labels']
        test_dataset_t = save['test_dataset']
        test_labels_t = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset_t.shape, train_labels_t.shape)
        print('Validation set', valid_dataset_t.shape, valid_labels_t.shape)
        print('Test set', test_dataset_t.shape, test_labels_t.shape)

    train_dataset_t, train_labels_t = reformat(train_dataset_t, train_labels_t)
    valid_dataset_t, valid_labels_t = reformat(valid_dataset_t, valid_labels_t)
    test_dataset_t, test_labels_t = reformat(test_dataset_t, test_labels_t)
    train_subset = 1000

    graph = tf.Graph()
    with graph.as_default():
        train_dataset = tf.constant(train_dataset_t[:train_subset, :], name="data_set")
        train_labels = tf.constant(train_labels_t[:train_subset], name="train_labels")
        tf.constant(valid_dataset_t, name="valid_dataset")
        tf.constant(valid_labels_t, name="valid_labels")

        num_class = 10  # train_labels.shape[1]
        size_of_inputs = 28 * 28  # train_dataset_t.shape[1]
        weights_t = tf.truncated_normal(shape=[size_of_inputs, num_class])
        biases_t = tf.zeros(shape=num_class)
        weights = tf.Variable(initial_value=weights_t, name='waights')
        biases = tf.Variable(initial_value=biases_t, name='bias')

        logistic_classifier = tf.matmul(train_dataset_t, weights) + biases
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logistic_classifier, labels=train_labels_t)
        min_loss = tf.reduce_mean(loss)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(min_loss)
        train_predictions = tf.nn.softmax(logistic_classifier)
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        for step in range(num_step):
            _, l, predictions = session.run([optimizer, min_loss, train_predictions])

        print "The end of learing."
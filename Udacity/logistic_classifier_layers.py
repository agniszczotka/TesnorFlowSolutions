import cPickle as pickle
import numpy as np

import tensorflow as tf

image_size = 28
num_class = 10
num_step = 1000


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


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
    batch_size = 128
    graph = tf.Graph()
    with graph.as_default():
        batch_dataset_holder = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size * image_size),
                                              name="data_set")
        batch_label_holder = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_class), name="train_labels")
        tf.constant(valid_dataset_t, name="valid_dataset")
        tf.constant(valid_labels_t, name="valid_labels")

        num_class = 10  # train_labels.shape[1]
        size_of_inputs = 28 * 28  # train_dataset_t.shape[1]
        nodes = 1024
        weights_t = tf.truncated_normal(shape=[size_of_inputs, nodes])
        biases_t = tf.zeros(shape=nodes)

        weights1 = tf.Variable(initial_value=weights_t, name='waights')
        biases1 = tf.Variable(initial_value=biases_t, name='bias')

        weights2 = tf.Variable(initial_value=tf.truncated_normal(shape=[nodes, num_class]))
        biases2 = tf.Variable(initial_value=tf.zeros(shape=num_class))

        hidden1 = tf.nn.relu(tf.matmul(train_dataset_t, weights1) + biases1)

        logistic_classifier = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logistic_classifier, labels=train_labels_t)
        min_loss = tf.reduce_mean(loss)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(min_loss)
        train_predictions = tf.nn.softmax(logistic_classifier)

        valid_predictions_hidden1 = tf.nn.relu(tf.matmul(valid_dataset_t, weights1) + biases1)
        valid_predictions = tf.nn.softmax(tf.nn.relu(tf.matmul(hidden1, weights2) + biases2))

        test_prediction_hidden1 = tf.nn.relu(tf.matmul(test_dataset_t, weights1) + biases1)
        test_prediction = tf.nn.softmax(
            tf.nn.relu(tf.matmul(test_prediction_hidden1, weights2) + biases2))

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        for step in range(num_step):

            offset = (step * batch_size) % (train_labels_t.shape[0] - batch_size)
            print offset

            # Generate a minibatch.
            batch_data = train_dataset_t[offset:(offset + batch_size), :]
            batch_labels = train_labels_t[offset:(offset + batch_size), :]
            feed_dic = {batch_dataset_holder: batch_data,
                        batch_label_holder: batch_labels}
            _, l, predictions = session.run([optimizer, min_loss, train_predictions], feed_dict=feed_dic)
            if step % 100 == 0:
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, batch_labels))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_predictions.eval(), valid_labels_t))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels_t))

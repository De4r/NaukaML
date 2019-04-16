import tensorflow as tf
import numpy as np
from DeepLearning.create_sentiment_featuresets import create_feature_sets_and_label
'''
input data -> weights -> hidden layer 1   (activation function) -> weights -> hidden layer 2 
(activation funkction) -> weights -> output layer
it its feed forward network
we compare output to intended output -> cost or loss function (cross entropy)
optimizer function -> minimaze cost (AdamOptimizer, SGD, AdaGrad)

backpropagation

feed forward + bracpropagation = epoch
'''

train_x, train_y, test_x, test_y = create_feature_sets_and_label('pos.txt', 'neg.txt')


n_nodes_hl1 = 500 # liczba neuronow w warstwie 1
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100 # 100 features - jeden batch

# height x width = 28*28=784 - flatten
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    #  (input_data * weights) + biases,  inicjacja wag i biasow
    # hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #
    # hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #
    # hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    #
    # output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    #                   'biases': tf.Variable(tf.random_normal([n_classes]))}
    # z komentarzy lepszy model
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    #  learining_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles ffw + epochs
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i <len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Acurracy:', accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)
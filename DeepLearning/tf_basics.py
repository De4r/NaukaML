import tensorflow as tf
import numpy as np


x1 = tf.constant([5])
x2 = tf.constant([6])

result = tf.multiply(x1, x2)
print(result)

sess = tf.Session()
print(sess.run(result))

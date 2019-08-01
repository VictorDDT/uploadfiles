import numpy as np
import tensorflow as tf
import datetime
from matplotlib import pyplot as plt
%matplotlib inline

#
# Preare data
#

N = 100
w_true = 6
b_true = 2
n = 300
x_data = np.random.rand(N, 1) * 1000
# print(x)
noise = np.random.normal(scale=n, size=(N, 1))
# print(noise)
y_data = w_true * x_data + b_true + noise
# print(y)
# y_data = np.reshape(y_data, (-1))
# print(y)

x_data = x_data.astype('float32')
y_data = y_data.astype('float32')

print(x_data.shape, y_data.shape)
print(x_data.dtype, y_data.dtype)
plt.scatter(x_data, y_data)

#
# Build and Traing the model
#

# placeholders
x = tf.placeholder(tf.float32, (N, 1), name='input_x')
y = tf.placeholder(tf.float32, (N, 1), name='input_y')

# weights
W = tf.Variable(tf.random_normal((1, 1)), name='var_W')
b = tf.Variable(tf.random_normal((1,)), name='var_b')

# prediction
y_pred = tf.matmul(x, W) + b

# loss
l = tf.reduce_sum((y - y_pred)**2)

# optimizer
train_op = tf.train.AdamOptimizer(.01).minimize(l)

# summary
tf.summary.scalar('loss', l)
tf.summary.scalar('W', tf.squeeze(W))
tf.summary.scalar('b', tf.squeeze(b))
merged = tf.summary.merge_all()

logPath = './logs/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_writer = tf.summary.FileWriter(logPath, tf.get_default_graph())

n_steps = 100000

# init session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # train model
    for i in range(n_steps):
        feed_dict = {x: x_data, y: y_data}
        _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
        print(f"step {i}, loss: {loss}")
        train_writer.add_summary(summary, i)
        
    sess.close()

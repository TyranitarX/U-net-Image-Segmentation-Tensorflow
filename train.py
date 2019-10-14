import tensorflow as tf
import loaddata
import factory


width = 256
length = 256
class_num = 1
iterTimes = 1000
Batch_size = 40
learning_rate = 1e-5

Data = loaddata.Unet_Data(True)

x = tf.placeholder(tf.float32, [None, width, length, 3])
y = tf.placeholder(tf.int32, [None, width, length])

y_ = factory.U_Net(x, class_num)

loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_))

tf.summary.scalar("loss", loss)
gradient = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(iterTimes):
    batch_data, batch_label = Data.NextBatch(5)
    loss_val, _ = sess.run(
        [loss, gradient],
        feed_dict={
            x: batch_data,
            y: batch_label
        }
    )
    print("iter %d , total loss %f" % (i, loss_val))

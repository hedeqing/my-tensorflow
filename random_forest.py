from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensor_forest.python import tensor_forest
mnist=input_data.read_data_sets('F:tensortflow\mnist.data',one_hot=True)
#import os
#os.environ("CUDA_VISIBLE_DEVICES")=""
#Parameters
num_steps=500
batch_size=1024
num_classes=10
num_features=784
max_nodes=1000
num_trees=10


X=tf.placeholder(tf.float32,shape=[None,num_features])
Y=tf.placeholder(tf.int32,shape=[None])
hparams=tensor_forest.ForestHParams(num_classes=num_classes,
                                    num_features=num_features,
                                    num_trees=num_trees,
                                    max_nodes=max_nodes).fill()
forest_graph=tensor_forest.RandomForestGraphs(hparams)
train_op=forest_graph.training_graph(X,Y)
loss_op=forest_graph.training_loss(X,Y)
infer_op=forest_graph.inference_graph(x)
correct_prediction=tf.equal(tf.argmax(infer_op,tf.int64))
accuracy_op=tf.reduce_mean(tf.cast(correct_predict,tf.float32))
init_vars=f.global_variables_initializer()
sess=tf.Session()
sess.run(init_vars)
for i in range(1,num_steps+1):
    batch_x,batch_y=mnist.train.next_bach(batch_size)
    _,i=ses.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
    print("step: %i,loss: %f,accc:%f"%(i,1,acc))
test_x,test_y=mnist.train.images,mnist.train.labels
print("test accuracy:",sess.run(accuracy_op,feed_dict={X:test_x,Y:test_y}))

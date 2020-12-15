
import tensorflow as tf

input = tf.keras.Input(shape=[3])
x = tf.keras.layers.Dense(20)(input)
x = tf.keras.layers.Dense(1)(x)
output = x

input_gt = tf.keras.Input(shape=[1], name="gtinput")
myloss = tf.reduce_mean(tf.math.square(input_gt - output), name="myloss")



m = tf.keras.Model(inputs=[input, input_gt], outputs=output)

m.add_loss(myloss)

m.compile()
m.summary()

import numpy as np 
a = np.random.randn(128,3)
a_true = np.random.randn(128,1)
m.fit( [a, a_true], epochs=10 )
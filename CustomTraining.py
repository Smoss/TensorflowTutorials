import tensorflow as tf

x = tf.zeros([10, 10])
x += 2
print(x)

v = tf.Variable(1.0)
assert v.numpy() == 1.0

v.assign(3.0)
assert v.numpy() == 3.0

class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

# plt.scatter(inputs, outputs, c='b')
# plt.scatter(inputs, model(inputs), c='r')
# plt.show()

print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())

@tf.function
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    dW, db = t.gradient(current_loss, [model.W, model.b])
    tf.print(dW)
    tf.print(db)
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(inputs))

    learning_rate = tf.constant(1. * .8 ** epoch)
    train(model, inputs, outputs, learning_rate)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()

from random import sample
import numpy as np
import argparse
import tensorflow as tf
from utils import *

tf.config.run_functions_eagerly(True)

## TODO: Remove used for plotting
def plot_metric(x, y, title, legends = []):
    import matplotlib.pyplot as plt
    if type(x) != type([]): x = [x]
    if type(y) != type([]): y = [y]
    assert(len(x) == len(y))
    colors = [ 'g-', 'r-', 'b-']
    for i in range(len(x)):
        plt.plot(x[i], y[i], colors[i % len(colors)])
    plt.legend(legends, loc='upper right')
    plt.title(title)
    plt.rcParams["figure.figsize"] = (30, 30)
    plt.grid()
    plt.xlim([np.min(x[:]), np.max(x[:])])
    plt.ylim([np.min(y[:])*.5, np.max(y[:])*1.5])
    plt.rc({'font.size': 42})
    plt.show()
####


class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        initializer = tf.keras.initializers.GlorotUniform()
        # in_size = 5
        self.internal_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, kernel_initializer=initializer, activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, kernel_initializer=initializer, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu'),
        ]
        self.layer_output = tf.keras.layers.Dense(out_size, kernel_initializer=initializer)
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?,|O|) tensor that keeps a batch of observations
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.layer_output(x)
        ########## Your code ends here ##########


def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: Remember, you can penalize steering (0th dimension) and throttle (1st dimension) unequally
    sample_weights = tf.constant(([0.7, 0.3]))
    y = y * sample_weights
    y_est = y_est * sample_weights
    return tf.reduce_mean(tf.square(y - y_est))
    # return tf.math.reduce_euclidean_norm(tf.y_est - y)
    # kl = tf.keras.losses.KLDivergence()
    # l = kl(y, y_est)
    # # return tf.reduce_all(l)
    # return l
    ########## Your code ends here ##########
    

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # tb_callback.set_model(nn_model)

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
        with tf.GradientTape() as tape:
            # forward pass
            y_est = nn_model(x, training=True) # use dropout
            # compute the loss
            current_loss = loss(y_est, y)
        grads = tape.gradient(current_loss, nn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, nn_model.trainable_variables))
        ########## Your code ends here ##########

        train_loss(current_loss)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)

    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    losses = []
    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch        
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
        losses.append(train_loss.result())


    #### TODO: REMOVE, USED for PLOTTING ####
    # for history in histories:
    length = len(losses)
    y = [ losses ]
    #  = history.history['accuracy']
    x  = [ np.linspace(0, length, length) ] 
        
    plot_metric(x, y, 'Training loss of CNNs using different loss functions', [ nn_model.name ])
    ###### 
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_IL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scenario', type=str, help="intersection, circularroad, lanechange", default="intersection")
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=5e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)

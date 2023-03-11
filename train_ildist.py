import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class MixtureDensityModelError(tf.keras.losses.Loss):

    def __init__(self, num_means, num_kernels, **kwargs):
        super(MixtureDensityModelError, self).__init__()
        self.num_means = num_means
        self.num_kernels = num_kernels
        self.z_alpha = None
        self.z_mu = None
        self.z_sigma = None

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        print(y_true.shape)
        print(y_pred.shape)
        offset = self.num_means * self.num_kernels
        self.z_mu = y_pred[:, :offset]
        self.z_alpha = y_pred[:, offset:offset+self.num_kernels]
        self.z_sigma = y_pred[:, offset+self.num_kernels:]

        E = tf.reduce_sum(self.compute_error(y_true, self.z_alpha, self.z_mu, self.z_sigma), 0)
        return -1 * tf.math.log(E)

    ######################################
    ####### internal functionality #######
    ######################################
        
    def compute_error(self, t, z_alpha, z_mu, z_sigma):
        '''
            Calculates a_i(x) * phi_i(t|x), for a single kernel (i.e. i = {1,2, OR ... , N})
        '''
        alpha = self.alpha(z_alpha)
        phi = self.phi(t, z_mu, z_sigma)
        
        return tf.transpose(alpha) *  phi

    def phi(self, t, z_mu, z_sigma):
        z_sigma = self.sigma(z_sigma)
        sigma = tf.linalg.diag(z_sigma) ** 2
        mu = z_mu
        print(sigma.shape)
        print(mu.shape)

        # TODO: Confirm below
        # gaussian pdf should single value
        # Note: there should be a phi per alpha
        # Should the gaussian pdf be a single value per kernel? Or two, as in each mu
        # Should there be 6 values for mu? As in `c` mu's
        # TODO: confirm, for each kernel: 
        #   mu.shape = (batch_size x num_kernels x c)
        #   sigma.shape = (batch_size x num_kernels x num_kernels)
        #   alpha.shape = (batch_size x num_kernels )
        mu = tf.reshape(mu, [mu.shape[0], self.num_means, self.num_kernels] )
        print(mu.shape)
        cholesky = tf.linalg.cholesky(sigma)
        print(cholesky.shape)
        probs = []
        for i in range(mu.shape[0]):
            mvn = tfd.MultivariateNormalTriL(loc=mu[i], scale_tril=cholesky[i])
            probs.append(mvn.prob(t[i]))
        # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        return tf.Variable(probs)

    def alpha(self, x):
        return tf.nn.softmax(x)

    def sigma(self, x):
        return tf.exp(x)

    ######################################
    ################ end ################
    ######################################

class MixtureDensityModelErrorOther(tf.keras.losses.Loss):

    def __init__(self, num_means, num_kernels, **kwargs):
        super(MixtureDensityModelError, self).__init__()
        self.num_means = num_means
        self.num_kernels = num_kernels
        self.z_alpha = None
        self.z_mu = None
        self.z_sigma = None

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        print(y_true.shape)
        print(y_pred.shape)
        offset = self.num_means * self.num_kernels
        self.z_mu = y_pred[:, :self.num_means]
        self.z_alpha = y_pred[:, offset:offset+self.num_kernels]
        self.z_sigma = y_pred[:, offset+self.num_kernels:]

        E = tf.reduce_sum(self.compute_error(y_true, self.z_alpha, self.z_mu, self.z_sigma), 0)
        return -1 * tf.math.log(E)

    ######################################
    ####### internal functionality #######
    ######################################
        
    def compute_error(self, t, z_alpha, z_mu, z_sigma):
        '''
            Calculates a_i(x) * phi_i(t|x), for a single kernel (i.e. i = {1,2, OR ... , N})
        '''
        alpha = self.alpha(z_alpha)
        phi = self.phi(t, z_mu, z_sigma)
        
        return tf.transpose(alpha) *  phi

    def phi(self, t, z_mu, z_sigma):
        z_sigma = self.sigma(z_sigma)
        sigma = tf.linalg.diag(z_sigma) ** 2
        mu = z_mu
        print(sigma.shape)
        print(mu.shape)
        mu = tf.reshape(mu, [mu.shape[0], self.num_means, self.num_kernels] )
        print(mu.shape)
        cholesky = tf.linalg.cholesky(sigma)
        print(cholesky.shape)
        probs = []
        for i in range(mu.shape[0]):
            mvn = tfd.MultivariateNormalTriL(loc=mu[i], scale_tril=cholesky[i])
            probs.append(mvn.prob(t[i]))
        # mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        return tf.Variable(probs)

    def alpha(self, x):
        return tf.nn.softmax(x)

    def sigma(self, x):
        return tf.exp(x)

    ######################################
    ################ end ################
    ######################################

class MixtureDensityModelErrorFinal(tf.keras.losses.Loss):

    def __init__(self, **kwargs):
        super(MixtureDensityModelErrorFinal, self).__init__()
        self.z_alpha = None
        self.z_mu = None
        self.z_sigma = None

    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        self.z_mu = y_pred[:, :2]
        self.z_sigma = y_pred[:, 2:]
        # print(self.z_mu.shape)
        # print(self.z_sigma.shape)
        B, N = self.z_sigma.shape
        self.z_sigma = tf.reshape(self.z_sigma, (B, int(N/2), int(N/2)))
        covariance = self.z_sigma @ tf.transpose(self.z_sigma, perm=[0, 2, 1])
        # print(covariance.shape)
        mvn = tfd.MultivariateNormalTriL(loc=self.z_mu, scale_tril=covariance)
        E = tf.reduce_mean(tf.math.log(mvn.prob(y_true)), 0)
        return -1 * E


class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        self.internal_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(24, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation='relu'),
            # tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(24, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(12, kernel_initializer=tf.keras.initializers.GlorotUniform(), activation='relu'),
        ]
        # num_outputs = (out_size + 2) * 3 # Removed after using only 6 ouputs
        num_outputs = out_size + 4
        self.layer_output = tf.keras.layers.Dense(num_outputs, kernel_initializer=tf.keras.initializers.GlorotUniform())
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?, |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        for i in range(len(self.internal_layers)):
            layer = self.internal_layers[i]
            x = layer(x)
        return self.layer_output(x)
        ########## Your code ends here ##########
   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, you can use MultivariateNormalFullCovariance or MultivariateNormalTriL, but they are not the only way.
    # loss_object = MixtureDensityModelError(num_means=2, num_kernels=3)
    loss_object = MixtureDensityModelErrorFinal()
    return loss_object(y, y_est)
    
    ########## Your code ends here ##########


def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

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

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
    nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)

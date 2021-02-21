import os, sys
sys.path.append(os.getcwd())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.plot

MODE = 'swgan-al'
# Valid options are: wgan, wgan-gp, swgan-gp, sgan-gp, wgan-al, swgan-al, sgan-al
SAMPLE_SIZE = 8 # How much gradients sampled between each pair of real data and fake data
RHO = 1e-4 # Quadratic weight penalty rho hyperparameter in Augmented Lagrangian Method
DATASET = '25gaussians' # 8gaussians, 25gaussians, swissroll
DIM = 512 # Model dimensionality
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 256 # Batch size
ITERS = 100000 # how many generator iterations to train for

lib.print_model_settings(locals().copy())

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples, real_data):
    if FIXED_GENERATOR:
        return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    else:
        noise = tf.random_normal([n_samples, 2])
        output = ReLULayer('Generator.1', 2, DIM, noise)
        output = ReLULayer('Generator.2', DIM, DIM, output)
        output = ReLULayer('Generator.3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator.4', DIM, 2, output)
        return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE, real_data)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# Loss function
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

if MODE != 'wgan':
    real_data_deep = tf.tile(real_data, [SAMPLE_SIZE, 1])
    fake_data_deep = tf.tile(fake_data, [SAMPLE_SIZE, 1])
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE * SAMPLE_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = alpha*real_data_deep + ((1-alpha)*fake_data_deep)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    if 'swgan' in MODE:
        gradients = tf.reshape(gradients, [BATCH_SIZE, SAMPLE_SIZE, 2])
        square_slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[2])
        average_square_slopes = tf.reduce_mean(square_slopes, reduction_indices=[1])
        omega = average_square_slopes
    elif 'wgan' in MODE:
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        omega = slopes
    elif 'sgan' in MODE:
        square_slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
        average_square_slopes = tf.reduce_mean(square_slopes)
        omega = average_square_slopes

    if 'swgan' in MODE:
        h_omega = tf.reduce_mean(tf.square(tf.nn.relu(omega - 1)))
    elif 'wgan' in MODE:
        h_omega = tf.reduce_mean(tf.square(omega - 1))
    elif 'sgan' in MODE:
        h_omega = omega - 1

    if 'gp' in MODE:
        if 'swgan' in MODE or 'sgan' in MODE:
            gradient_penalty = LAMBDA * (h_omega**2)
        else:
            gradient_penalty = LAMBDA * h_omega
    elif 'al' in MODE:
        alpha = lib.param('alpha', tf.zeros(shape=tf.shape(h_omega)))
        gradient_penalty = alpha*h_omega+0.5*RHO*(h_omega**2)
    
    disc_cost += gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE != 'wgan':
    if 'gp' in MODE:
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            disc_cost, 
            var_list=disc_params
        )
    elif 'al' in MODE:
        disc_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        disc_train_op = disc_opt.apply_gradients(disc_gv)
        alpha_opt = tf.train.GradientDescentOptimizer(learning_rate=-RHO)
        alpha_gv = alpha_opt.compute_gradients(disc_cost, var_list=lib.params_with_name('alpha'))
        alpha_train_op = alpha_opt.apply_gradients(alpha_gv)
    
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

print "Generator params:"
for var in lib.params_with_name('Generator'):
    print "\t{}\t{}".format(var.name, var.get_shape())
print "Discriminator params:"
for var in lib.params_with_name('Discriminator'):
    print "\t{}\t{}".format(var.name, var.get_shape())

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run(
        [fake_data, disc_real], 
        feed_dict={real_data:points}
    )
    disc_map = session.run(disc_real, feed_dict={real_data:points})

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.contourf(x, y, disc_map.reshape((len(x), len(y))).transpose(), alpha=0.1)
    # plt.colorbar()

    plt.scatter(true_dist[:, 0], true_dist[:, 1], s=10, c='red', marker='o')
    plt.scatter(samples[:, 0], samples[:, 1], s=10, c='black', marker='o')
    plt.xticks(())
    plt.yticks(())

    plt.savefig('frame'+str(frame_index[0])+'.pdf')
    frame_index[0] += 1

# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':
    
        dataset = []
        for i in xrange(100000/25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in xrange(len(dataset)/BATCH_SIZE):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE, 
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield data

    elif DATASET == '8gaussians':
    
        scale = 2.
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in xrange(BATCH_SIZE):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield dataset

# Train loop!
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        for i in xrange(CRITIC_ITERS):
            if 'al' in MODE:
                _data = gen.next()
                _disc_cost, _, __ = session.run(
                    [disc_cost, disc_train_op, alpha_train_op],
                    feed_dict={real_data: _data}
                )
            else:    
                _data = gen.next()
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])
        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        if iteration % 100 == 99:
            lib.plot.flush()
            generate_image(_data)
        lib.plot.tick()

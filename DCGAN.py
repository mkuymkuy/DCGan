import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

batch_size = 64
z_dimension = 128

lower_bound = 0.01
upper_bound = 0.01

def leaky_relu(x,alpha = 0.2):
    return tf.maximum(x, alpha * x)

def batch_norm(input,isTraining,decay = 0.999):
    #this is for conv only
    epsilon = 1e-5
    shape = input.get_shape().as_list()
    scale = tf.get_variable('scale',[shape[3]],initializer = tf.constant_initializer(1))
    offset = tf.get_variable('offset',[shape[3]],initializer = tf.constant_initializer(0))
    
    pop_mean = tf.get_variable('pop_mean',[shape[3]],initializer = tf.constant_initializer(0),trainable = False)
    pop_var = tf.get_variable('pop_var',[shape[3]],initializer = tf.constant_initializer(1),trainable = False)
    
    if isTraining:
        batch_mean, batch_var = tf.nn.moments(input,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input,
                batch_mean, batch_var, offset, scale, epsilon)
    
    return tf.nn.batch_normalization(input,
                pop_mean, pop_var, offset, scale, epsilon)

def conv2d(input,kernal_shape,isTraining):
    weights = tf.get_variable('weights',kernal_shape,initializer = tf.variance_scaling_initializer())
    #xavier initializer set scale factor as 2 for relu
    #biases = tf.get_variable('biases',bias_shape,initializer = tf.constant_initializer(0))
    #comment out biases since batch_norm is applied
    
    output = tf.nn.conv2d(input = input, filter = weights, strides = [1,1,1,1], padding = 'SAME')
    output = batch_norm(output,isTraining)
    output = leaky_relu(output)
    output = tf.nn.avg_pool(output,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    return output

def deconv2d(input,kernal_shape,output_shape,isTraining,norm = True):
    weights_de = tf.get_variable('weights_de',kernal_shape,initializer = tf.variance_scaling_initializer())
    
    output = tf.nn.conv2d_transpose(value = input, filter = weights_de,output_shape = output_shape,strides = [1,2,2,1], padding = 'SAME')
    if norm:
        output = batch_norm(output,isTraining)
    
    return output

def fcn(input,input_size,output_size,isTraining,norm = True):
    wf = tf.get_variable('wf',[input_size,output_size],initializer = tf.variance_scaling_initializer())
    if norm:
        output = batch_norm(tf.matmul(input,wf),isTraining)
    else:
        bf = tf.get_variable('bf',[output_size],initializer = tf.constant_initializer(0))
        output = tf.matmul(input,wf) + bf
        
    return output

def discriminator(images,ifReuse = False,isTraining = True):
    with tf.variable_scope('discriminator',reuse = ifReuse):
        with tf.variable_scope('conv1'):
            d = conv2d(images,[5,5,1,32],isTraining)
        with tf.variable_scope('conv2'):
            d = conv2d(d,[5,5,32,64],isTraining)
        with tf.variable_scope('fcnd1'):
            d = tf.reshape(d,[-1,7 * 7 * 64])
            d = fcn(d,7 * 7 * 64,1024,isTraining,norm = False)
            d = leaky_relu(d)
        with tf.variable_scope('fcnd2'):
            d = fcn(d,1024,1,isTraining,norm = False)
            p = tf.sigmoid(d)
    return d,p

def generator(input,input_dimension,batch_size = 64,isTraining = True):
    with tf.variable_scope('generator'):
        with tf.variable_scope('fcng1'):
            d = fcn(input,input_dimension,7 * 7 * 64,isTraining,norm = False)
            d = leaky_relu(d)
            d = tf.reshape(d,[-1,7,7,64])
        with tf.variable_scope('deconv1'):
            d = deconv2d(d,[5,5,32,64],[batch_size,14,14,32],isTraining)
            d = leaky_relu(d)
        with tf.variable_scope('deconv2'):
            d = deconv2d(d,[5,5,1,32],[batch_size,28,28,1],isTraining,norm = False)
            d = tf.tanh(d)
            
    return d

tf.reset_default_graph()

z_input = tf.placeholder(tf.float32, [None,z_dimension])
d_input = tf.placeholder(tf.float32, [None,28,28,1])

#generate a batch of fake images
z_output = generator(z_input,z_dimension)
#judge the real images
d_output_real,p_real = discriminator(d_input)
#judge the fake images
d_output_fake,p_fake = discriminator(z_output,ifReuse = True)

mode = 'WGan'
if mode == 'vanilla':
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_output_real, labels = tf.ones_like(d_output_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_output_fake, labels = tf.zeros_like(d_output_fake)))

    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_output_fake, labels = tf.ones_like(d_output_fake)))
else:
    d_loss = tf.reduce_mean(d_output_fake - d_output_real)
    g_loss = tf.reduce_mean(-d_output_fake)

d_correct_real = p_real > 0.5
d_correct_fake = p_fake > 0.5

accuracy_real = tf.reduce_mean(tf.cast(d_correct_real,tf.float32))
accuracy_fake = tf.reduce_mean(tf.cast(d_correct_fake,tf.float32))

tvars = tf.trainable_variables()

gvars = [var for var in tvars if 'generator' in var.name]
dvars = [var for var in tvars if 'discriminator' in var.name]

if mode == 'vanilla':
    d_trainer = tf.train.GradientDescentOptimizer(0.001).minimize(d_loss, var_list=dvars)
    g_trainer = tf.train.AdamOptimizer(0.0003).minimize(g_loss, var_list=gvars)
else:
    d_trainer = tf.train.RMSPropOptimizer(0.0003).minimize(d_loss, var_list=dvars)
    g_trainer = tf.train.RMSPropOptimizer(0.001).minimize(g_loss, var_list=gvars)
	with tf.control_dependencies([d_trainer]):
        clip = (tf.tuple([tf.assign(var, tf.clip_by_value(var, lower_bound, upper_bound)) for var in dvars]))

#provide infos for TensorBoard
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss)
tf.summary.scalar('real_accuracy', accuracy_real)
tf.summary.scalar('fake_accuracy', accuracy_fake)

images_for_tensorboard = generator(z_input, z_dimension,batch_size,False)
tf.summary.image('Generated_images', images_for_tensorboard, 5)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir)

saver = tf.train.Saver()

#start training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer.add_graph(sess.graph)
print(datetime.datetime.now())
for i in range(100000):
    #normalize images data from [0,1] to [-1,1]
    real_images = (mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1]) - 0.5) * 2
    z_noise = np.random.normal(-1,1,[batch_size,z_dimension])
    # Train discriminator on both real and fake images
    _, ac_real,ac_fake = sess.run([d_trainer, accuracy_real, accuracy_fake],
                                           {z_input: z_noise, d_input: real_images})

    # Train generator
    z_noise = np.random.normal(-1,1,[batch_size,z_dimension])
    _ = sess.run(g_trainer, feed_dict={z_input: z_noise})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_noise = np.random.normal(-1,1,[batch_size,z_dimension])
        summary = sess.run(merged, {z_input: z_noise, d_input: real_images})
        writer.add_summary(summary, i)

    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_noise = np.random.normal(-1,1,[1,z_dimension])
        generated_images = generator(z_input, z_dimension,1,False)
        images = sess.run(generated_images, {z_input: z_noise})
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        plt.show()

print(datetime.datetime.now())
#save the model


save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in file: %s" % save_path)

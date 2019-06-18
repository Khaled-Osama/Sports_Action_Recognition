'''
this file *Main* is implemented to train the model and save it.
If the model is already trained and saved then it will load it and calculate the accuracy of validation set.

'''

# Main
from I3d import *
from Data_utils import *
from Hyperparameters import *
import os
from tqdm import tqdm

tf.reset_default_graph()
'''
loading the training and validation dataset.
'''
train_dataset, train_labels, train_frames = load_training_data()
valid_dataset, valid_labels, valid_frames = load_validation_data()

train_video = tf.placeholder(tf.float32, shape=(1, None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
train_label = tf.placeholder(tf.int32, shape=(BATCH_SIZE, num_of_classes))

'''
create an instance of the model and path to it a training example to calculate the output of the final layer(logits)
'''
with tf.variable_scope('RGB'):
    rgb_model = InceptionI3D()
    train_logits = rgb_model(train_video, is_training=True, dropout_keep_prob=dropout_keep_prob)

# convert our variables to mach with checkpoint variables
rgb_variable_map = {}
for variable in tf.global_variables():
    if variable.name[:len('RGB/inception_i3d/Logits/')] == 'RGB/inception_i3d/Logits/':
        continue
    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable

# specify trainable variables(last layer only)
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               "RGB/inception_i3d/Logits")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits, labels=train_label))

valid_video = tf.placeholder(tf.float32, shape=(1, None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
valid_label = tf.placeholder(tf.int32, shape=(None, num_of_classes))
valid_logitss = tf.placeholder(tf.float32, shape=(None, num_of_classes))

valid_logits = rgb_model(valid_video, is_training=False)
valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logitss,
                                                                       labels=valid_label))
optimizer = tf.train.AdamOptimizer(learning_rate)

train_op = optimizer.minimize(loss, var_list=train_vars)
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

with tf.variable_scope('accuracy'):
    valid_correct_preds = tf.equal(tf.argmax(valid_logitss, 1), tf.argmax(valid_label, 1))
    valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_preds, tf.float32))

    train_correct_preds = tf.equal(tf.argmax(train_logits, 1), tf.argmax(train_label, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_correct_preds, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''
    the model is is already trained and saved in that path.
    so, we can load its weights and run on the validatin data to measure the accuracy. 
    '''
    if os.path.exists('i3d'):
        saver.restore(sess, model_checkpoint)
        _valid_logits = np.zeros((num_of_validation, 10))

        print('Model is exist, now we calculating the validation accuracy.')

        for i in tqdm(range(num_of_validation)):
            _valid_video = read_and_preprocess_video(valid_dataset[i], valid_frames[i])
            _valid_logits[i] = sess.run(valid_logits, feed_dict={valid_video: _valid_video})
        valid_acc = sess.run(valid_accuracy, feed_dict={valid_logitss: _valid_logits,
                                                        valid_label: valid_labels})
        print('validation accuracy = ' + str(valid_acc))
    else:
        '''
        loading the weights of the pretrained model (kinetics model)
        and apply the transfer learning.
        '''
        rgb_saver.restore(sess, kinetics_checkpoint)

        # Training
        k = 0
        for epoch in range(NUM_EPOCHS):

            for step in range(num_of_training_example):
                print('step =' + str(step))

                _train_video = read_and_preprocess_video(train_dataset[step], train_frames[step])

                sess.run(train_op, feed_dict={train_video: _train_video,
                                              train_label: train_labels[step, np.newaxis]})

                '''
                every 10 example (10 updates) measure the error and accuracy of a batch from validation set,
                the batch_size =6
                
                
                '''
                if step > 0 and step % 10 == 0:
                    batch_valid_logits = np.zeros((6, 10))
                    for i in range(VALID_BATCH_SIZE):
                        k %= 138
                        _valid_video = read_and_preprocess_video(valid_dataset[k], valid_frames[k])
                        batch_valid_logits[i] = sess.run(valid_logits, feed_dict={valid_video: _valid_video})
                        k += 1
                    _valid_loss, _valid_acc = sess.run([valid_loss, valid_accuracy], feed_dict={
                        valid_logitss: batch_valid_logits,
                        valid_label: valid_labels[k - 6:k]})

                    _train_loss, _train_acc = sess.run([loss, train_accuracy], feed_dict={train_video: _train_video,
                                                                                          train_label: train_labels[
                                                                                              step, np.newaxis]})

                    print('train_error  =  ' + str(_train_loss))
                    print('train_acc  =  ' + str(_train_acc))

                    print('valid_error  =  ' + str(_valid_loss))

                    print('valid_accuracy  =  ' + str(_valid_acc))
                else:
                    '''
                    monitoring each accuracy and loss for each training example.
                    '''
                    _train_loss, _train_acc = sess.run([loss, train_accuracy], feed_dict={train_video: _train_video,
                                                                                          train_label: train_labels[
                                                                                              step, np.newaxis]})
                    print('train_error  =  ' + str(_train_loss))
                    print('train_acc  =  ' + str(_train_acc))
            # after each epoch save the wiegths
            saver.save(sess, model_checkpoint)
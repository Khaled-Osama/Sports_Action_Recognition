'''
this file *testing* is implemented to load and test the model over the whole testing dataset,
then draw the conf_matrix.

'''
import seaborn as sn
from Data_utils import *
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from I3d import *
'''
this function takes the actual labels & preictions.
by using seaborn library it visualize the confusion matrix in good way.
'''
def build_conf_matrix(labels, predictions):
  matrix = tf.math.confusion_matrix(labels, predictions)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mat = sess.run(matrix)
    df = pd.DataFrame(mat, index = [i for i in label_mapping.keys()], columns = [i for i in label_mapping.keys()])
    plt.figure(figsize = (10,7))
    sn.heatmap(df, annot=True)

'''
loading the testing set
'''
test_dataset, test_labels, test_frames = load_testing_data()
test_video = tf.placeholder(tf.float32, shape=(1, None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

with tf.variable_scope('RGB'):
  rgb_model = InceptionI3D()
test_logits = rgb_model(test_video, is_training=False)

test_logitss = tf.placeholder(tf.float32, shape=(num_of_testing_example, 10))
'''
calculate the accuracy.
'''
with tf.variable_scope('accuracy'):
  test_correct_preds = tf.equal(tf.argmax(test_logitss, 1), tf.argmax(test_labels, 1))
  test_accuracy = tf.reduce_mean(tf.cast(test_correct_preds, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # restore the wights of the model.
  saver.restore(sess, 'my_model/model.ckpt')

  _test_logits = np.zeros((num_of_testing_example, 10))
  '''
  this loop iterate over testing set and load one by one and calculate its logits.
  '''
  for i in tqdm(range(num_of_testing_example)):
    # print(i)
    _test_video = read_and_preprocess_video(test_dataset[i], test_frames[i])
    _test_logits[i] = sess.run(test_logits, feed_dict={test_video: _test_video})
  acc = sess.run(test_accuracy, feed_dict={test_logitss: _test_logits})

  print(acc)

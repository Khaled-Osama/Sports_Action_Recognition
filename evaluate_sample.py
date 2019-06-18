'''
this file *evaluate_sample* is responsible for predict the query input(video)

'''
from I3d import *
from Data_utils import *

'''
Call this function
inputs:
video -> numpy array of dimension -> (1, num_of_frames, 224, 224, 3)
'''

def get_query_prediction(video_path):

    '''
    convert video path to video frames.
    '''
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    num_frames = 0
    while success:
        success, frame = cap.read()
        num_frames += 1

    video_frames = np.zeros((1, num_frames, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    i = 0
    while success:
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = frame / 255.0
            frame = frame * 2
            frame -= 1
            video_frames[0][i] = frame
            i += 1

    with tf.variable_scope('RGB'):
        model = InceptionI3D()
        query_video = tf.placeholder(tf.float32, shape=(1, None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
        query_logits = model(query_video, is_training=False)
        query_logits = tf.nn.softmax(query_logits)
    restorer = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, model_checkpoint)

        _query_logits = sess.run(query_logits, feed_dict={query_video:video_frames})
        query_pred = class_names[np.argmax(_query_logits)]
    return query_pred


path = os.getcwd()
path += '/video_example'

path = pathlib.Path(path)
path = str(path)
path += '/*.jpg'

#video = read_and_preprocess_video('video_example/2538-11_70015.avi')
print(get_query_prediction('video_example/7608-5_70039.avi'))

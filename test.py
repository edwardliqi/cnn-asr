import tensorflow as tf
import numpy as np
import librosa
import asr
def test(source):
    wave, sr = librosa.load(source, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    print(np.array(mfcc).shape)
    mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
    mfcc = np.array(mfcc)
    with tf.Session() as sess:
        width = 20  # mfcc features
        height = 80  # (max) length of utterance
        classes = 10  # digits
    
        config = asr.CNNConfig
        cnn = asr.ASRCNN(config, width, height, classes)
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint('cnn_model')
        saver.restore(sess, model_file)
        print("get cnn_model success")
        feed_dict = {cnn.input_x: [mfcc], cnn.keep_prob: 1.0 }
        cls = sess.run(cnn.y_pred_cls, feed_dict=feed_dict)
        #y = np.argmax(prediction[0])
        print(cls)
if __name__ == '__main__':
    print('start')
    test('data/spoken_numbers_pcm/9_Alex_260.wav')

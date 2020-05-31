from model import *
from data_import import *
from scipy.io import wavfile
from time import sleep


class SEInferenceWrapper:
    # SPEECH ENHANCEMENT NETWORK
    __SE_LAYERS = 13  # NUMBER OF INTERNAL LAYERS
    __SE_CHANNELS = 64  # NUMBER OF FEATURE CHANNELS PER LAYER
    __SE_LOSS_LAYERS = 6  # NUMBER OF FEATURE LOSS LAYERS
    __SE_NORM = "NM"  # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

    __fs = 16000

    def __init__(self, model_dir_path):
        assert os.path.isdir(model_dir_path)
        self.__model_file_path = os.path.join(model_dir_path, 'se_model.ckpt')

        # SET LOSS FUNCTIONS AND PLACEHOLDERS
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.variable_scope(tf.get_variable_scope()):
            input = tf.placeholder(tf.float32,shape=[None,1,None,1])
            clean = tf.placeholder(tf.float32,shape=[None,1,None,1])

            self.__enhanced = senet(input, n_layers=self.__SE_LAYERS, norm_type=self.__SE_NORM, n_channels=self.__SE_CHANNELS)

        # INITIALIZE GPU CONFIG
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.__sess = tf.Session(config=config)

        print("Config ready")

        self.__sess.run(tf.global_variables_initializer())

        print("Session initialized")

        saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
        saver.restore(self.__sess, self.__model_file_path)


    @staticmethod
    def __get_in_file_tensor(input_file_path):
        assert os.path.isfile(input_file_path)

        fs, inputData = wavfile.read(input_file_path)
        inputData = np.reshape(inputData, [-1, 1])
        shape = np.shape(inputData)
        inputData = np.reshape(inputData, [1, 1, shape[0], shape[1]])

        return inputData

    def denoise_file(self, input_file_path, out_file_path):
        # # SET LOSS FUNCTIONS AND PLACEHOLDERS
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        #     input = tf.placeholder(tf.float32,shape=[None,1,None,1])
        #     clean = tf.placeholder(tf.float32,shape=[None,1,None,1])
        #
        #     enhanced = senet(input, n_layers=self.__SE_LAYERS, norm_type=self.__SE_NORM, n_channels=self.__SE_CHANNELS)
        #
        # # INITIALIZE GPU CONFIG
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        #
        # print("Config ready")
        #
        # sess.run(tf.global_variables_initializer())
        #
        # print("Session initialized")
        #
        # saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
        # saver.restore(sess, self.__model_file_path)
        inputData = self.__get_in_file_tensor(input_file_path)

        # Session run
        output = self.__sess.run([self.__enhanced],
                            feed_dict={input: inputData})
        output = np.reshape(output, -1)
        wavfile.write(out_file_path, self.__fs, output)


if __name__ == '__main__':
    in_file_path = 'test_recs/1.wav'
    out_file_path = 'test_recs_denoised/1.wav'
    model_dir = 'models'

    wrapper = SEInferenceWrapper(model_dir)
    print(f'Wait for 5s')
    sleep(5)
    wrapper.denoise_file(in_file_path, out_file_path)
    assert os.path.isfile(out_file_path)

from model import *
from data_import import *
from scipy.io import wavfile
import argparse


def main(in_file_path, out_file_path):
    # SPEECH ENHANCEMENT NETWORK
    SE_LAYERS = 13  # NUMBER OF INTERNAL LAYERS
    SE_CHANNELS = 64  # NUMBER OF FEATURE CHANNELS PER LAYER
    SE_LOSS_LAYERS = 6  # NUMBER OF FEATURE LOSS LAYERS
    SE_NORM = "NM"  # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

    fs = 16000

    modfolder = "models"

    # SET LOSS FUNCTIONS AND PLACEHOLDERS
    with tf.variable_scope(tf.get_variable_scope()):
        input = tf.placeholder(tf.float32, shape=[None, 1, None, 1])
        clean = tf.placeholder(tf.float32, shape=[None, 1, None, 1])

        enhanced = senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

    # INITIALIZE GPU CONFIG
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("Config ready")
    sess.run(tf.global_variables_initializer())
    print("Session initialized")

    saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
    model_file_path = os.path.join(modfolder, 'se_model.ckpt')
    saver.restore(sess, model_file_path)

    fs, inputData = wavfile.read(in_file_path)
    inputData = np.reshape(inputData, [-1, 1])
    shape = np.shape(inputData)
    inputData = np.reshape(inputData, [1, 1, shape[0], shape[1]])

    # VALIDATION ITERATION
    output = sess.run([enhanced],
                      feed_dict={input: inputData})
    output = np.reshape(output, -1)
    wavfile.write(out_file_path, fs, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform DFL denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--in_file", type=str, required=True, help= \
        "Path to src file (file to denoise).")
    parser.add_argument("-o", "--out_file", type=str, required=True, help= \
        "Path to denoised (processed) file.")
    args = parser.parse_args()

    print(f'In file: {args.in_file}')
    print(f'Out file: {args.out_file}')

    assert os.path.isfile(args.in_file)

    main(args.in_file, args.out_file)

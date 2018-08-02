import functools

from loader import *
from ops import *


def doublewrap(function):

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


# This is a decorator. Which is itself decorated. Okay. It says that the wrapped function will only be executed
# once. Okay. So I do not want to run the wrapped functions, which in this case are tensorflow ops many times. Okay.
# Also it says that subsequent calls will directly return the result, so that the operations are added to the graph
# only once. Hmm. This will take some time to understand. There is more. If this decorator is used with arguments, they
# will be forwarded to the variable scope. It also says that the operations added by the function live within a
# tf.variable_scope(). The scope name defaults to the name of the wrapped function. Okay this somewhat makes sense.
@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:
    def __init__(self, image, label, dropout=0.5, conv_size=9, conv_stride=1, ksize=2, pool_stride=2, filter_num=128,
                 padding="SAME"):

        self.image = image
        self.label = label
        self.dropout = dropout
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.ksize = ksize
        self.pool_stride = pool_stride
        self.padding = padding
        self.filter_num = filter_num

        # the following three are required for tf.train.Saver()
        self.prediction
        self.optimize
        self.accuracy

    @define_scope
    def prediction(self):
        with tf.variable_scope("model") as scope:
            #input image
            input_image = self.image

            layers = []

            # conv_1 [batch, ngf, 5] => [batch, 64, ngf]
            with tf.variable_scope("conv_1"):
                output = relu(conv1d(input_image, self.filter_num, name='conv_1'))
                layers.append(output)

            # conv_2 - conv_6
            layer_specs = [
                (self.filter_num * 2, 0.5),  # conv_2: [batch, 64, ngf] => [batch, 32, ngf * 2]
                (self.filter_num * 4, 0.5),  # conv_3: [batch, 32, ngf * 2] => [batch, 16, ngf * 4]
                (self.filter_num * 8, 0.5),  # conv_4: [batch, 16, ngf * 4] => [batch, 8, ngf * 8]
                (self.filter_num * 8, 0.5),  # conv_5: [batch, 8, ngf * 8] => [batch, 4, ngf * 8]
                (self.filter_num * 8, 0.5)  # conv_6: [batch, 4, ngf * 8] => [batch, 2, ngf * 8]
            ]

            # adding layers
            for _, (out_channels, dropout) in enumerate(layer_specs):
                with tf.variable_scope("conv_%d" % (len(layers) + 1)):
                    rectified = lrelu(layers[-1], 0.2)

                    # [batch, in_width, in_channels] => [batch, in_width/2, out_channels]
                    convolved = conv1d(rectified, out_channels)

                    # batchnormalize convolved
                    output = batchnorm(convolved, is_2d=False)

                    # dropout
                    if dropout > 0.0:
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    layers.append(output)

            #fc1
            h_fc1 = relu(fully_connected(layers[-1], 256, name='fc1'))

            #dropout
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout)

            #fc2
            result = tf.sigmoid(fully_connected(h_fc1_drop, 2, name='fc2'))

            return result

    @define_scope
    def optimize(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.prediction))
        return tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    @define_scope
    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():
    # Import data
    db = load_stock_data("data/daily/")

    # Construct graph
    image = tf.placeholder(tf.float32, [None, 128, 5])
    label = tf.placeholder(tf.float32, [None, 2])
    dropout = tf.placeholder(tf.float32)
    model = Model(image, label, dropout=dropout)

    # Saver
    saver = tf.train.Saver()

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):  # was 500000 here before
            images, labels = db.train.next_batch(10)  # was 10 here before
            if i % 10 == 0:  # was 100 here before
                images_eval, labels_eval = db.test.next_batch(1000)  # was 1000 here before
                accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
                print('step %d, accuracy %g' % (i, accuracy))
            sess.run(model.optimize, {image: images, label: labels, dropout: 0.5})
            if i % 25 == 0:  # was 10000 here before
                save_path = 'checkpoints/'
                model_name = 'stocks_model.ckpt'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path_full = os.path.join(save_path, model_name)
                saver.save(sess, save_path_full, global_step=i+1)
        images_eval, labels_eval = db.test.next_batch(1000)  # was 1000 here before
        accuracy = sess.run(model.accuracy, {image: images_eval, label: labels_eval, dropout: 1.0})
        print('final accuracy on testing set: %g' % (accuracy))
    print("finished")


if __name__ == '__main__':
    main()

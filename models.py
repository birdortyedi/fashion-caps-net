from keras import backend as K
from keras import layers, models, initializers
from keras.utils import multi_gpu_model
from utils import squash
import tensorflow as tf


class MultiGPUNet(models.Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(MultiGPUNet, self).__getattribute__(attrname)


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(Length, self).get_config()


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            mask = K.one_hot(indices=K.argmax(K.sqrt(K.sum(K.square(inputs), -1)), 1),
                             num_classes=K.sqrt(K.sum(K.square(inputs), -1)).get_shape().as_list()[1])

        return K.batch_flatten(inputs * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        return super(Mask, self).get_config()


def PrimaryCaps(inputs, dim_capsule, n_channels=32, kernel_size=9, strides=2, padding="same"):
    inputs = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    inputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(inputs)
    return layers.Lambda(squash, name='primarycap_squash')(inputs)


class FashionCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(FashionCaps, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        inputs = K.expand_dims(inputs, 1)
        inputs = K.tile(inputs, [1, self.num_capsule, 1, 1])
        inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs)

        # Dynamic routing
        b = tf.zeros(shape=[K.shape(inputs)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0
        for i in range(self.routings):
            outputs = squash(K.batch_dot(tf.nn.softmax(b, dim=1), inputs, [2, 2]))

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(FashionCaps, self).get_config()
        new_config = list(base_config.items()) + list(config.items())
        return dict(new_config)


def FashionCapsNet(input_shape, n_class, args):
    x = layers.Input(shape=input_shape)

    # Stacked-convolutional layer 1
    out = layers.Conv2D(filters=int(args.conv_filters/4),
                        kernel_size=args.conv_kernel_size,
                        strides=1,
                        padding='valid',
                        name='conv1')(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)

    # Stacked-convolutional layer 2
    out = layers.Conv2D(filters=int(args.conv_filters/2),
                        kernel_size=args.conv_kernel_size,
                        strides=2,
                        padding='valid',
                        name='conv2')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)

    # Stacked-convolutional layer 3
    out = layers.Conv2D(filters=args.conv_filters,
                        kernel_size=args.conv_kernel_size,
                        strides=2,
                        padding='valid',
                        name='conv3')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)

    # Stacked-convolutional layer 4
    out = layers.Conv2D(filters=args.conv_filters,
                        kernel_size=args.conv_kernel_size,
                        strides=2,
                        padding='valid',
                        name='conv4')(out)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)

    # PrimaryCaps
    out = PrimaryCaps(out, dim_capsule=args.dim_capsule)

    # FashionCaps
    out = FashionCaps(num_capsule=n_class, dim_capsule=args.dim_capsule,
                      routings=args.routings, name='fashioncaps')(out)

    # Length of each capsule represents the probability of the existence of the entity
    out_caps = Length(name='capsnet')(out)

    # Mask the output of FashionCapsNet
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([out, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(out)  # Mask using the capsule with maximal length. For prediction

    # Transpose-convolutional decoder network for reconstruction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(7*7*args.conv_filters, activation='relu', input_dim=args.dim_capsule*n_class))
    decoder.add(layers.Reshape((7, 7, args.conv_filters)))
    decoder.add(layers.Conv2DTranspose(args.conv_filters, kernel_size=args.conv_kernel_size, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters/2), kernel_size=args.conv_kernel_size, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters / 4), kernel_size=args.conv_kernel_size, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters / 8), kernel_size=args.conv_kernel_size, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(3, kernel_size=9, strides=2, padding='same', activation='sigmoid'))

    # Model for training
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    # Model for evaluation
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model

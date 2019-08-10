from keras import layers, models
from keras.utils import multi_gpu_model
from layers import Mask, Length, PrimaryCaps, FashionCaps


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

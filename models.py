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

    def residual_block(y, nb_channels, _strides=(2, 2), _project_shortcut=False):
        shortcut = y

        # down-sampling is performed with a stride of 2
        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

        return y

    # Stacked-convolutional layer 1
    out = layers.Conv2D(filters=32, kernel_size=7, strides=1, padding='same', name='conv1_1')(x)
    out = layers.BatchNormalization()(out)
    out = layers.LeakyReLU()(out)
    out = layers.SpatialDropout2D(rate=0.3)(out)

    out = residual_block(out, nb_channels=64, _project_shortcut=True)
    out = layers.SpatialDropout2D(rate=0.3)(out)
    out = residual_block(out, nb_channels=128, _project_shortcut=True)
    out = layers.SpatialDropout2D(rate=0.3)(out)
    out = residual_block(out, nb_channels=256, _project_shortcut=True)
    out = layers.SpatialDropout2D(rate=0.3)(out)

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
    decoder.add(layers.Conv2DTranspose(args.conv_filters, kernel_size=4, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters/2), kernel_size=4, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters / 4), kernel_size=4, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(int(args.conv_filters / 8), kernel_size=4, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid'))

    # Model for training
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    # Model for evaluation
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model

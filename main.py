"""
Author: Furkan Kınlı
Version: 2.0
E-mail: furkan.kinli@ozu.edu.tr

Base code for capsule architecture:
XifengGuo - Github: https://github.com/XifengGuo/CapsNet-Keras

FashionCapsNet: Clothing Classification with Capsule Networks
Keras (Backend: TF) implementation of FashionCapsNet
Dataset: DeepFashion (290K training, 40K validation and 40K test images
                        with 46 fine-grained category labels for clothing images)
Run:
        python main.py --args

Training:
        Validation accuracy converges at 255. epoch
        Apprx. 15 days to complete train on multi-gpu of GTX1080Ti
Result:
        Test accuracy:
                        Top-1: 63.61%
                        Top-3: 83.18%
                        Top-5: 89.83%
"""
import os
from utils import margin_loss, top_3_categorical_accuracy, custom_generator, get_iterator, save_recons
from config import get_arguments
from models import FashionCapsNet, MultiGPUNet
from keras import backend as K
from keras import optimizers, callbacks


def train(model, args):
    # Define callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv', append=True)
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=args.verbose)
    early_stopper = callbacks.EarlyStopping(monitor='val_capsnet_loss', patience=args.patience, verbose=args.verbose)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': ['accuracy', top_3_categorical_accuracy, 'top_k_categorical_accuracy']})

    # Start training using custom generator
    model.fit_generator(generator=custom_generator(get_iterator(args.filepath,
                                                                args.input_size,
                                                                args.shift_fraction,
                                                                args.hor_flip,
                                                                args.whitening,
                                                                args.rotation_range,
                                                                args.brightness_range,
                                                                args.shear_range,
                                                                args.zoom_range,
                                                                subset="train"),
                                                   testing=args.testing),
                        steps_per_epoch=int(210000 / args.batch_size),
                        epochs=args.epochs,
                        validation_data=custom_generator(get_iterator(args.filepath, subset="val"),
                                                         testing=args.testing),
                        validation_steps=int(40000 / args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay, early_stopper],
                        initial_epoch=args.initial_epoch)

    # Save the model
    model_path = '/t_model.h5'
    model.save(args.save_dir + model_path)
    print('The model saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': ['accuracy', top_3_categorical_accuracy, 'top_k_categorical_accuracy']})

    # Evaluate the model using custom generator
    scores = model.evaluate_generator(generator=custom_generator(get_iterator(args.filepath,
                                                                              subset="test"),
                                                                 testing=args.testing),
                                      steps=int(40000 / args.batch_size))
    print(scores)

    # Reconstruct batch of images
    if args.recon:
        x_test_batch, y_test_batch = get_iterator(args.filepath, subset="test").next()
        y_pred, x_recon = model.predict(x_test_batch)

        # Save reconstructed and original images
        save_recons(x_recon, x_test_batch, y_pred, y_test_batch, args.save_dir)


if __name__ == '__main__':
    K.clear_session()
    args = get_arguments()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model, eval_model = FashionCapsNet(input_shape=(args.input_size, args.input_size, 3),
                                       n_class=46,
                                       args=args)

    if args.weights is not None:
        model.load_weights(args.weights, by_name=True)

    model.summary()

    if args.multi_gpu:
        p_model = MultiGPUNet(model, args.multi_gpu)
        # p_eval_model = MultiGPUNet(eval_model, args.multi_gpu)

    if not args.testing:
        if args.multi_gpu:
            train(model=p_model, args=args)
            # implicitly sure that p_model defined
        else:
            train(model=model, args=args)
    else:
        if args.weights is None:
            print('Random initialization of weights.')
        test(model=eval_model, args=args)

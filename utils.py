import os
import cv2
import numpy as np
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing import image


class DirectoryIteratorWithBoundingBoxes(image.DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['origin']['x'], bounding_box['origin']['y'],
                     bounding_box['width'],bounding_box['height']],
                    dtype=K.floatx())

        # optionally save augmented images to disk for debugging purposes
        # if self.save_to_dir:
        #     for i in range(len(batch_x)):
        #         img = image.array_to_img(batch_x[i], self.data_format, scale=True)
        #         fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
        #                                                           index=current_index + i,
        #                                                           hash=np.random.randint(1e4),
        #                                                           format=self.save_format)
        #         img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y


def margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 0.1
    l = 0.5

    L = y_true * K.square(K.maximum(0., m_plus - y_pred)) + \
        l * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus))

    return K.mean(K.sum(L, 1))


def squash(activations, axis=-1):
    scale = K.sum(K.square(activations), axis, keepdims=True) / \
            (1 + K.sum(K.square(activations), axis, keepdims=True)) / \
            K.sqrt(K.sum(K.square(activations), axis, keepdims=True) + K.epsilon())
    return scale * activations


def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def custom_generator(iterator, testing=True):
    if testing:
        while True:
            x_batch, y_batch = iterator.next()
            yield (x_batch, [y_batch, x_batch])
    else:
        while True:
            x_batch, y_batch = iterator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])


def get_iterator(file_path, input_size=224, shift_fraction=0., h_flip=False, zca_whit=False, rot_range=0.,
                 bright_range=0., shear_range=0., zoom_range=0., subset="train"):

    file_path = os.path.join(file_path, subset)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if subset == "train":
        data_gen = image.ImageDataGenerator(width_shift_range=shift_fraction,
                                            height_shift_range=shift_fraction,
                                            horizontal_flip=h_flip,
                                            zca_whitening=zca_whit,
                                            rotation_range=rot_range,
                                            brightness_range=bright_range,
                                            shear_range=shear_range,
                                            zoom_range=zoom_range,
                                            rescale=1./255)
    else:
        data_gen = image.ImageDataGenerator(rescale=1./255)

    t_iterator = DirectoryIteratorWithBoundingBoxes(file_path, data_gen, target_size=(input_size, input_size))
    return t_iterator


def get_labels_dict():
    labels_dict = {
        0: 'Anorak', 1: 'Blazer', 2: 'Blouse', 3: 'Bomber', 4: 'Button-Down', 5: 'Caftan', 6: 'Capris',
        7: 'Cardigan', 8: 'Chinos', 9: 'Coat', 10: 'Coverup', 11: 'Culottes', 12: 'Cutoffs', 13: 'Dress',
        14: 'Flannel', 15: 'Gauchos', 16: 'Halter', 17: 'Henley', 18: 'Hoodie', 19: 'Jacket', 20: 'Jeans',
        21: 'Jeggings', 22: 'Jersey', 23: 'Jodhpurs', 24: 'Joggers', 25: 'Jumpsuit', 26: 'Kaftan', 27: 'Kimono',
        28: 'Leggings', 29: 'Onesie', 30: 'Parka', 31: 'Peacoat', 32: 'Poncho', 33: 'Robe', 34: 'Romper',
        35: 'Sarong', 36: 'Shorts', 37: 'Skirt', 38: 'Sweater', 39: 'Sweatpants', 40: 'Sweatshorts', 41: 'Tank',
        42: 'Tee', 43: 'Top', 44: 'Trunks', 45: 'Turtleneck'
    }
    return labels_dict


def save_recons(x_recon, x_test_batch, y_pred, y_test_batch, save_dir):
    labels_dict = get_labels_dict()
    recon_dir = os.path.join(save_dir, 'recon_real_imgs')
    file_extension = ".jpg"

    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)

    for i, (recon, test) in enumerate(zip(x_recon, x_test_batch)):
        img = cv2.cvtColor(np.uint8((recon * 255.)), cv2.COLOR_RGB2BGR)
        real = cv2.cvtColor(np.uint8((test * 255.)), cv2.COLOR_RGB2BGR)

        recon_filename = recon_dir + '/' + str(i) + '_recon_' + labels_dict[np.argmax(y_test_batch[i])] + \
                         '_' + labels_dict[np.argmax(y_pred[i])] + file_extension
        real_filename = recon_dir + '/' + str(i) + '_real_' + labels_dict[np.argmax(y_test_batch[i])] + \
                        '_' + labels_dict[np.argmax(y_pred[i])] + file_extension

        print(recon_filename)
        print(real_filename)

        cv2.imwrite(recon_filename, img)
        cv2.imwrite(real_filename, real)

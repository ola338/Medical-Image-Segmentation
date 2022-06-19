import cv2
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from augmentators import Augmentators
from AU_Net import att_unet
import glob
import itertools


def train_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_split))
            ids_train_batch = train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread(train_img_path_template.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))

                for lst in itertools.product([False, True], repeat=5):
                    _img, _mask = Augmentators.augment(img, mask,
                                                       force=True,
                                                       hue=lst[0],
                                                       scale=lst[1],
                                                       sharp=lst[2],
                                                       hflip=lst[3],
                                                       vflip=lst[4])

                    _mask = np.expand_dims(_mask, axis=2)
                    x_batch.append(_img)
                    y_batch.append(_mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(valid_split))
            ids_valid_batch = valid_split[start:end]
            for id in ids_valid_batch:
                img = cv2.imread(train_img_path_template.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


if __name__ == '__main__':
    epochs = 50
    batch_size = 1

    input_size, model = att_unet()
    # model.load_weights(filepath= #'weights/att_unet_weights/best_au_weights.hdf5') # For resuming train

    train_img_path_template = 'dataset/train/x-ray_st/{}.png'
    train_img_mask_path_template = 'dataset/train/mask_manual/{}.png'

    train_filenames = glob.glob("dataset/train/x-ray_st/*.png")
    train_filenames = [filename.replace('\\', '/').replace('.png', '') for filename in train_filenames]
    train_filenames = [filename.split('/')[-1] for filename in train_filenames]

    train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

    filepath_unet = 'weights/unet_weights/best_u_weights.hdf5'
    filepath_att_unet = 'weights/att_unet_weights/best_au_weights.hdf5'

    callbacks = [
        #        EarlyStopping(monitor='val_dice_loss',
        #                           patience=8,
        #                           verbose=1,
        #                           min_delta=1e-4,
        #                           mode='max'),
        ReduceLROnPlateau(monitor='val_dice_loss',
                          factor=0.5,
                          patience=4,
                          verbose=1,
                          epsilon=1e-5,
                          mode='max'),
        ModelCheckpoint(monitor='val_dice_loss',
                        filepath=filepath_att_unet,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='max'),
        TensorBoard(log_dir='logs')]

    print(f'Training on {len(train_split)} samples')
    print(f'Validating on {len(valid_split)} samples')

    model.fit_generator(generator=train_generator(),
                        steps_per_epoch=np.ceil(float(len(train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(),
                        validation_steps=np.ceil(float(len(valid_split)) / float(batch_size)))

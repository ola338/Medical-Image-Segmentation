import cv2
import numpy as np
from tqdm import tqdm
from u_net import get_unet_128
import glob
from AU_Net import att_unet

if __name__ == '__main__':
    orig_width = 240
    orig_height = 320

    threshold = 0.5

    epochs = 10
    batch_size = 1
    input_size, model = att_unet()

    test_img_path_template = 'dataset/test/x-ray_st/{}.png'
    test_img_mask_path_template = 'dataset/test/mask_manual/{}.png'

    filepath_unet = 'weights/unet_weights/best_u_weights.hdf5'
    filepath_att_unet = 'weights/att_unet_weights/best_au_weights.hdf5'
    model.load_weights(filepath=filepath_att_unet)

    print(input_size)

    test_filenames = glob.glob("dataset/test/x-ray_st/*.png")
    test_filenames = [filename.replace('\\', '/').replace('.png', '') for filename in test_filenames]
    test_filenames = [filename.split('/')[-1] for filename in test_filenames]

    print('Predicting on {} samples with batch_size = {}...'.format(len(test_filenames), batch_size))
    for start in tqdm(range(0, len(test_filenames), batch_size)):
        x_batch = []
        end = min(start + batch_size, len(test_filenames))
        ids_test_batch = test_filenames[start:end]
        for id in ids_test_batch:
            img = cv2.imread(test_img_path_template.format(id))
            img = cv2.resize(img, (input_size, input_size))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for index, pred in enumerate(preds):
            prob = np.array(cv2.resize(pred, (orig_width, orig_height)) > threshold).astype(np.float32) * 255
            current_filename = ids_test_batch[index]
            # print(bce_dice_loss(x_batch, prob))
            cv2.imwrite(f'dataset/test/seg_pred_att/{id}.png', prob)

    print("Done!")

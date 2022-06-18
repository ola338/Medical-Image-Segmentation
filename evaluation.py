import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import  f1_score, jaccard_score
from AU_Net import att_unet


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.int32)
    return x

if __name__ == "__main__":
    
    filepath_att_unet = 'weights/att_unet_weights/best_au_weights.hdf5'  
    input_size, model = att_unet()
    model.load_weights(filepath=filepath_att_unet) 

    test_x = sorted(glob("dataset/test/seg_pred_att/*"))
    test_y = sorted(glob("dataset/test/mask_manual/*"))

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        image_name = x.split("/")[-1]

        x = read_image(x)
        y = read_mask(y)

        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        y = y.flatten()
        y_pred = y_pred.flatten()

        dice_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([image_name, dice_value, jac_value])

    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Dice: {score[0]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
   
    df = pd.DataFrame(SCORE)
    df.columns = ["Image", "Dice", "Jaccard"]
    df.to_csv("score-seg_pred_att_finall.csv")

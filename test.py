import os
from src.utils.HED_data_parser import DataParser
from src.networks.hed import hed
from keras.utils import plot_model
from keras import backend as K
from keras import callbacks
import numpy as np
import glob
from PIL import Image
import cv2

test = glob.glob('images/*')

if __name__ == "__main__":
    #environment
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    # if not os.path.isdir(model_dir): os.makedirs(model_dir)
    # model
    model = hed()
    # plot_model(model, to_file=os.path.join(model_dir, 'model.pdf'), show_shapes=True)

    # training
    # call backs
    model.load_weights('./checkpoints/HEDSeg/checkpoint.212-0.11.hdf5')
    # train_history = model.predict()
    for image in test:
        name = image.split('/')[-1]
        x_batch = []
        im = Image.open(image)
        (h,w) = im.size
        print (h,w)
        im = im.resize((480,480))
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
        x_batch = np.array(x_batch, np.float32)
        prediction = model.predict(x_batch)
        mask = np.zeros_like(im[:,:,0])
        for i in range(len(prediction)):
            mask += np.reshape(prediction[i],(480,480))
        ret,mask = cv2.threshold(mask,np.mean(mask)+1.2*np.std(mask),255,cv2.THRESH_BINARY)
        out_mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_CUBIC)
        # out_mask = mask.resize((h,w))
        cv2.imwrite("output/%s" % name, out_mask)
        # out_img = Image.fromarray(mask, astype='float32').resize((h,w))
        # out_img.save('./b.jpg')

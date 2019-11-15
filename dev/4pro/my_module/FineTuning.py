import numpy as np
import cv2
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model

class FT:

    def __init__(self):
        # VGG16環境の構築
        self.model = VGG16(
        include_top = True,      #1000クラスに分類するフル結合層を含むかどうか．
        weights = "imagenet",    #重みの種類：ImageNetを使って学習した重み
        input_tensor = None,     #モデル画像を入力する場合に使用
        input_shape = None       #入力画像の形状を指定：(224, 224, 3)
        )
        # model.summary()    #モデルの構造を表示

    def prepro_img(self, img):
        img = img.resize(self.model.input_shape[1:3])  #モデルの入力サイズでリサイズする
        x = image.img_to_array(img)   #PIL.Imageオブジェクトをnp.float32型のnumpy配列に変換
        x = np.expand_dims(x, axis=0) #配列の形状を(Height, Width, Channels)から(1, Height, Width, Channels)に変更
        x = preprocess_input(x)       #VGG16用の前処理を行う
        return x

    # 特徴量の抽出:欲しい層の番号を入れる
    def get_layer_output(self, img, layer_num):
        get_layer = K.function([self.model.layers[0].input],[self.model.layers[layer_num].output])
        preds_result = get_layer([img,0])[0]
        print('特徴量のサイズ : {}'. format(preds_result.shape))
        resize_result = self.resize_bilinear(preds_result)
        print('リサイズ後の特徴量のサイズ : {}'. format(resize_result.shape))
        return resize_result

    '''
    def feature_print(self, preds_result):
        print('特徴量のサイズ : {}'. format(preds_result.shape))

        result_x = decode_predictions(preds_x, top=3)[0]
        for _, name, score in result_x:
            print('{}: {:.2%}'.format(name, score))
        '''

    # バイリニア法にてリサイズ
    def resize_bilinear(self, src):
        src = np.squeeze(src)
        dst = np.zeros((src.T.shape[0], 224, 224))
        for i in range(src.T.shape[0]):
            dst[i] = cv2.resize(src.T[i], (224, 224), interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(dst.T, axis=0)
        return x

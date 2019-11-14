import numpy as np
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

    # 特徴量の抽出
    def get_feature(self, img):
        x = self.prepro_img(img)
        preds_x = self.model.predict(x)  #推論を行う
        return preds_x

    def feature_print(self, preds_x):
        print('特徴量のサイズ : {}'. format(preds_x.shape))
        result_x = decode_predictions(preds_x, top=3)[0]
        for _, name, score in result_x:
            print('{}: {:.2%}'.format(name, score))


'''
# 特徴量の出力
# 中間層ver
layer_name = 'activation_4'
hidden_layer_model = Model(
inputs=model.input,
outputs=model.get_layer(layer_name).output
)
# 最後の層ver
intermediante_layer_model = Model(
inputs=model.input,
outputs=model.get_layer("block5_pool").output
)
'''

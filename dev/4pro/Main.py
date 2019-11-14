import cv2
import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from my_module import keras_YOLO, FineTuning


input_path = 0       #カメラの設定
output_path =  ""    #映像の保存先
yolo = keras_YOLO.YOLO()
ft = FineTuning.FT();

def main():

    img_path = input('Input image filename:')
    i = 1
    try:
        pre_img = Image.open(img_path)
        img = Image.open(img_path)
    except:
        print('Open Error!')
    else:
        obj_results = yolo.get_objs(img)
        for obj in obj_results:
            print('======{}枚目======'.format(i))
            img = img.crop((obj[1], obj[2], obj[3], obj[4]))
            feature = ft.get_feature(img)
            ft.feature_print(feature)
            i = i+1

def camera_main():

    # 映像の設定　
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))             #コーデック(FourCC)の特定
    video_fps    = vid.get(cv2.CAP_PROP_FPS)                     #FPSの取得
    video_size   = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) #横幅・縦幅の取得

    '''
    # 動画を保存したい時に設定
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    '''

    # FPSの計算の設定
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        stop_key = cv2.waitKey(1)&0xff      #カメラを止めるキーの設定(Qで止まる)
        return_value, frame = vid.read()    #動画の読み込み：返り値は成功かどうかのbool値と画像の配列
        image = Image.fromarray(frame)      #numpy配列からPIL Imageへ変更
        vid_cap = np.asarray(image)          #リストからnumpy配列へ変更

        # FPSの計算
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        # FPSの表示
        cv2.putText(vid_cap, text = fps, org = (20, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 2, color = (255, 255, 255), thickness = 2, lineType = cv2.LINE_AA)

        # 動画の表示
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.imshow("Video", vid_cap)

        '''動画を保存する時はコメントアウトを外す
        if isOutput:
            out.write(result)
        '''
        # スペースキーが押されたらyoloにかける
        if stop_key == 32:
            obj_results = yolo.get_objs(image)
            for obj in obj_results:
                img = image.crop((obj[1], obj[2], obj[3], obj[4]))
                feature = ft.get_feature(img)

        # エスケープキーが押されたらカメラを止める
        if stop_key == 27:
            exit()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()

#ui所需的库
import cv2 as cv
import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk

#运行函数所需的库
import numpy as np
import jieba
import data_input_helper as data_helpers
import argparse
import os.path
import re
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf

#综合图像识别和文字分类
import os
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *
sys.path.append('textcnn')
from classify_image import *

#忽略警告
import warnings
warnings.filterwarnings("ignore")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#垃圾分类函数
class RafuseRecognize():

    def __init__(self):
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt',
                                      model_dir='/tmp/imagenet')

    def init_classify_image_model(self):
        create_graph('/tmp/imagenet')

        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')

    def recognize_image(self, image_data):
        predictions = self.sess.run(self.softmax_tensor,
                                    {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)
            # print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))
            # print(human_string)
            classification = self.refuse_classification.predict(human_string)
            result_list.append('%s  =>  %s' % (human_string, classification))

        return '\n'.join(result_list)
        
        
        
        
        
#新建窗口
window = tkinter.Tk()
window.title('垃圾分类识别界面')
window.geometry('350x400')

#设置显示文本
tkinter.Label(window, text='请输入文本: ', font=("微软雅黑", 18)).place(x=30, y=50)
tkinter.Label(window, text='文本识别结果：', font=("微软雅黑", 18)).place(x=30, y=150)

tkinter.Label(window, text='图片路径：', font=("微软雅黑", 18)).place(x=30, y=250)
tkinter.Label(window, text='图片识别结果: ', font=("微软雅黑", 18)).place(x=30, y=350)

#基本设置
#var_user_name = tkinter.StringVar()
a1= tkinter.StringVar()
entry = tkinter.Entry(window, textvariable=a1, font=("微软雅黑", 15))
entry.place(x=200, y=60, width=300, height=30)
a2= tkinter.StringVar()
entry2 = tkinter.Entry(window, textvariable=a2, font=("微软雅黑", 15))
entry2.place(x=200, y=160, width=300, height=30)

load = tkinter.StringVar()
entry_jiazai = tkinter.Entry(window, textvariable=load, font=("微软雅黑", 15))
entry_jiazai.place(x=200, y=260, width=400, height=30)

a3 = tkinter.StringVar()
entry3 = tkinter.Entry(window, textvariable=a3, font=("微软雅黑", 15))
entry3.place(x=200, y=360, width=400, height=30)
FLAGS = None


# 打开文件函数
def file():
    chooseFile = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    load.set(chooseFile)

def text():
    test = RefuseClassification()
    res = test.predict(entry2.get())
    a2.set(res)
    
    
# 识别图片数字函数
def image(img):
    form = RafuseRecognize()
    picture= tf.gfile.FastGFile(img, 'rb').read()
    res = form.recognize_image(picture)
    a3.set(res)



# 按钮
submit_button = tkinter.Button(window, text="文本识别",font=("微软雅黑", 20), command=lambda: text()).place(x=770, y=100)
submit_button = tkinter.Button(window, text="选择文件", font=("微软雅黑", 20), command=file).place(x=770, y=200)
submit_button = tkinter.Button(window, text="图片识别", font=("微软雅黑", 20),command=lambda: image(load.get())).place(x=770, y=300)
window.mainloop()
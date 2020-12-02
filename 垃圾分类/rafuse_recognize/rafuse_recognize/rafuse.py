
#用于整合imagenet分类模型、textcnn映射模型

#导入库和predict中的垃圾分类的类
import numpy as np
import os, sys
sys.path.append('textcnn')
from textcnn.predict import RefuseClassification
from classify_image import *

#垃圾分辨函数
class RafuseRecognize():
    
    def __init__(self):
        
        self.refuse_classification = RefuseClassification()
        self.init_classify_image_model()
        #节点的查找
        self.node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', 
                                model_dir = '/tmp/imagenet')
        
     #初始化分类图像模型   
    def init_classify_image_model(self):
        #创建计算图
        create_graph('/tmp/imagenet')
        #创建会话
        self.sess = tf.Session()
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        
     #分辨图片的函数，垃圾分类   
    def recognize_image(self, image_data):
        
        predictions = self.sess.run(self.softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)#删除一层维度

        top_k = predictions.argsort()[-5:][::-1]
        result_list = []
        for node_id in top_k:
            human_string = self.node_lookup.id_to_string(node_id)#转化位string字符串
            #print(human_string)
            human_string = ''.join(list(set(human_string.replace('，', ',').split(','))))
            #print(human_string)
            classification = self.refuse_classification.predict(human_string)#分类结果
            result_list.append('%s  =>  %s' % (human_string, classification))#将结果添加进列表
            
        return '\n'.join(result_list)#返回换行后的数据
        
#主函数
#垃圾分辨函数调用，图片导入，分类结果输出
if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RafuseRecognize()
        image_data = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        res = test.recognize_image(image_data)
        print('classify:\n%s' %(res))

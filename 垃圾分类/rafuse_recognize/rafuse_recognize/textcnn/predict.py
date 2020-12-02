#导入库与其他py文件
import tensorflow as tf
import numpy as np
import os, sys
import data_input_helper as data_helpers
import jieba

#flag解析
# Parameters参数处理
# Data Parameters 数据参数

tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")
#tf.flags来定义参数
#DEFINE_string()限定了可选参数输入必须是string
#总的来说就是定义一个用于接收 string 类型数值的变量
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
#同理DEFINE_integer定义一个用于接收 int 类型数值的变量，DEFINE_boolean() : 定义一个用于接收 bool 类型数值的变量;
# Eval Parameters 评估参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS   #从对应的命令行参数取出参数
#FLAGS._parse_flags()
FLAGS.flag_values_dict() #将其解析成字典存储到FLAGS.__flags中


#垃圾分类的类
class RefuseClassification():

    def __init__(self):
    
        self.w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)  #加载词向量
        self.init_model()
        self.refuse_classification_map = {0: '可回收垃圾', 1: '有害垃圾', 2: '湿垃圾', 3: '干垃圾'}
        
     #处理数据的函数   
    def deal_data(self, text, max_document_length = 10):
        
        words = jieba.cut(text)
        x_text = [' '.join(words)]
        x = data_helpers.get_text_idx(x_text, self.w2v_wr.model.vocab_hash, max_document_length)

        return x

    #inceptionv3图像分类网络结构的解析与代码实现
    def init_model(self):
        #tf.train.latest_checkpoint()函数的作用查找最新保存的checkpoint文件的文件名
        #checkpoint_dir保存变量的目录。
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        
        graph = tf.Graph()    # tf.Graph()函数创建计算图
        with graph.as_default():  #定义属于计算图graph的张量和操作
            
            #tf.ConfigProto()配置tf.Session的运算方式，比如gpu运算或者cpu运算
            #allow_soft_placement允许动态分配GPU内存
            #log_device_placement打印出设备信息
            session_conf = tf.ConfigProto(
                              allow_soft_placement=FLAGS.allow_soft_placement, 
                              log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            self.sess.as_default()
            # Load the saved meta graph and restore variables
            #加载保存的meta图并恢复变量
            #tf.train.import_meta_graph用来加载meta文件中的图,以及图上定义的结点参数包括权重偏置项等需要训练的参数,也包括训练过程生成的中间参数
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))#加载训练好的模型的meta值
            saver.restore(self.sess, checkpoint_file)

            # Get the placeholders from the graph by name
            #从图表中按名称获取占位符
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
          
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            #我们要计算的tensors
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                
    #预测
    def predict(self, text):
    
        x_test = self.deal_data(text, 5)
        predictions = self.sess.run(self.predictions, {self.input_x: x_test, self.dropout_keep_prob: 1.0})
        
        refuse_text = self.refuse_classification_map[predictions[0]]
        return refuse_text

#输出结果
if __name__ == "__main__":
    if len(sys.argv) == 2:
        test = RefuseClassification()
        res = test.predict(sys.argv[1])
        print('classify:', res)

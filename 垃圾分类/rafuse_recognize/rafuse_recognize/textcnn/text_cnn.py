#coding=utf-8
import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    用于文本分类的CNN。
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    使用一个嵌入层，然后是卷积层、最大池层和softmax层
    """
    def __init__(
      self,w2v_model, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        #输入、输出和退出的占位符
        #tf.placeholder定义一个变量和维度，tf.int32和tf.float32指变量类型，None不限维度，sequence_length表示列数
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #记录l2正则化损失（可选）
        l2_loss = tf.constant(0.0)#创建一个常量

        # Embedding layer 嵌入层
        with tf.device('/cpu:0'), tf.name_scope("embedding"): #指定用cpu运算,tf.name_scope()用来让变量共享
            if w2v_model is None:
            #tf.Variable() 每次都会产生新的变量
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="word_embeddings")
            else:
            #tf.get_variable() 如果遇到了已经存在名字的变量时, 
            #它会单纯的提取这个同样名字的变量，如果不存在名字的变量再创建.
                self.W = tf.get_variable("word_embeddings",
                    initializer=w2v_model.vectors.astype(np.float32))

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # tf.nn.embedding_lookup查找索引的值
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)#tf.expand_dims增加一个维度

        # Create a convolution + maxpool layer for each filter size
        #为每个过滤器大小创建一个卷积+maxpool层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity 应用非线性
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs 最大池超过输出
                #tf.nn.max_pool()池化操作形成一维向量
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features 合并所有池功能
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout 添加dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        #最终（未标准化）分数和预测
        with tf.name_scope("output"):
           W = tf.get_variable(
                        "W",
                        shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
           b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
           l2_loss += tf.nn.l2_loss(b)         
           self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
           self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        #计算交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy 准确度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


  

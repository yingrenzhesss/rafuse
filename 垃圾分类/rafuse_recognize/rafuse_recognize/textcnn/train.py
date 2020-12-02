#coding=utf-8
#导入库和py文件
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_input_helper as data_helpers
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn

# Parameters参数
#tf.flags来定义参数
#DEFINE_string()限定了可选参数输入必须是string
#同理DEFINE_integer定义一个用于接收 int 类型数值的变量，DEFINE_boolean() : 定义一个用于接收 bool 类型数值的变量;

# Data loading params数据加载参数
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "./data/train_data.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
tf.flags.DEFINE_string("w2v_file", "./data/word2vec.bin", "w2v_file path")
# Model Hyperparameters模型超参数
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2, 3, 4", "Comma-separated filter sizes (default: '3, 4, 5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters训练参数
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



#加载数据
def load_data(w2v_model):
    """Loads starter word-vectors and train/dev/test data."""
    #“加载启动词向量和训练/开发/测试数据。
    
    # Load the starter word vectors加载起始词向量
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.train_data_file)

    max_document_length = max([len(x.split(" ")) for x in x_text])
    print ('len(x) = ', len(x_text), ' ', len(y))
    print(' max_document_length = ', max_document_length)

    x = []
    vocab_size = 0
    if(w2v_model is None):
       #learn.preprocessing.VocabularyProcessor(max_document_length)
       #根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
       #max_document_length 最大文档长度
      vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
      
      #从x_text中学习到一个词汇表并返回一个id矩阵
      x = np.array(list(vocab_processor.fit_transform(x_text)))
      vocab_size = len(vocab_processor.vocabulary_)

      # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(int(time.time()))))
      vocab_processor.save("vocab.txt")
      print( 'save vocab.txt')
    else:
      x = data_helpers.get_text_idx(x_text, w2v_model.vocab_hash, max_document_length)
      vocab_size = len(w2v_model.vocab_hash)
      print('use w2v .bin')


    #索引值处理
    #训练集和测试集的获取
    np.random.seed(10)#设定一个随机数种子
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    return x_train, x_dev, y_train, y_dev, vocab_size#返回训练集和测试集，还有词向量大小

#训练模型
def train(w2v_model):
    # Training
    x_train, x_dev, y_train, y_dev, vocab_size= load_data(w2v_model)
    with tf.Graph().as_default(): #返回值：返回一个上下文管理器，这个上下管理器使用这个图作为默认的图
         #tf.ConfigProto()配置tf.Session的运算方式，比如gpu运算或者cpu运算
         #allow_soft_placement允许动态分配GPU内存
         #log_device_placement打印出设备信息
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement, 
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        #分类算法TextCNN，主要思想是将不同长度的短文作为矩阵输入，
        #使用多个不同size的filter去提取句子中的关键信息，并用于最终的分类
        with sess.as_default():
            cnn = TextCNN(
                w2v_model, 
                sequence_length=x_train.shape[1], 
                num_classes=y_train.shape[1], 
                vocab_size=vocab_size, 
                embedding_size=FLAGS.embedding_dim, 
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(", "))), 
                num_filters=FLAGS.num_filters, 
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure 确定训练程序
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            #跟踪渐变值和稀疏度（可选）
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            #模型和摘要的输出目录
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            print("Writing to {}\n".format(out_dir))


            # Summaries for loss and accuracy
            #生成准确率和损失率标量图
            
            loss_summary = tf.summary.scalar("loss", cnn.loss) #tf.summary.scalar()用来显示标量信息，
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries训练总结
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])#保存信息
            train_summary_dir = os.path.join(out_dir, "summaries", "train")#文件路径
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph) #指定一个文件用来保存图。
                                                                                         #下面同理
            # Dev summaries开发总结
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            ##检查Checkpoint目录。Tensorflow假设这个目录已经存在，所以我们需要创建它
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)#tf.train.Saver()保存和加载模型

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables初始化所有变量
            #含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，
            #也就是在debug时还不是一个tensor量，而是一个Variable变量类型
            sess.run(tf.global_variables_initializer())


            #训练步骤
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch, 
                  cnn.input_y: y_batch, 
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy, (w, idx) = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.get_w2v_W()], 
                #     feed_dict)
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], 
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print w[:2], idx[:2]
                train_summary_writer.add_summary(summaries, step)

            #开发步骤
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                评估开发集上的模型
                """
                feed_dict = {
                  cnn.input_x: x_batch, 
                  cnn.input_y: y_batch, 
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy], 
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches生成批处理
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            #测试步骤
            def dev_test():
                batches_dev = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                for batch_dev in batches_dev:
                    x_batch_dev, y_batch_dev = zip(*batch_dev)
                    dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)

            # Training loop. For each batch...每个批次循环训练
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)# #获得global_step
                # Training loop. For each batch...
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_test()


                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

#主函数，执行模型训练
if __name__ == "__main__":  
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    train(w2v_wr.model)

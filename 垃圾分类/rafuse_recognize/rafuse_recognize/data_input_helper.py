import numpy as np
import re
import word2vec
import jieba

class w2v_wrapper:
     def __init__(self, file_path):
        # w2v_file = os.path.join(base_path,  "vectors_poem.bin")
        self.model = word2vec.load(file_path)
        if 'unknown' not  in self.model.vocab_hash:
            unknown_vec = np.random.uniform(-0.1, 0.1, size=128) #生成120个-0.1到0.1的数
            self.model.vocab_hash['unknown'] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors, unknown_vec))# np.row_stack水平拼接两个参数

#处理数据，将匹配到的字符串替换
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    除SST外的所有数据集的标记化/字符串清理
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(), !?\'\`]",  " ",  string)
    string = re.sub(r"\'s",  " \'s",  string)
    string = re.sub(r"\'ve",  " \'ve",  string)
    string = re.sub(r"n\'t",  " n\'t",  string)
    string = re.sub(r"\'re",  " \'re",  string)
    string = re.sub(r"\'d",  " \'d",  string)
    string = re.sub(r"\'ll",  " \'ll",  string)
    string = re.sub(r", ",  " ,  ",  string)
    string = re.sub(r"!",  " ! ",  string)
    string = re.sub(r"\(",  " \( ",  string)
    string = re.sub(r"\)",  " \) ",  string)
    string = re.sub(r"\?",  " \? ",  string)
    string = re.sub(r"\s{2, }",  " ",  string)
    return string.strip().lower()           #返回处理后的字符串，将其小写并删除前后空格


#提取非零索引值
def removezero( x,  y):
    nozero = np.nonzero(y)  # np.nonzero提取非零的索引值
    print('removezero', np.shape(nozero)[-1], len(y)) #返回提取后的长度和原长度

    if(np.shape(nozero)[-1] == len(y)):
        return np.array(x), np.array(y)

    y = np.array(y)[nozero]
    x = np.array(x)
    x = x[nozero]
    return x,  y


#将文件转换为一个列表输出的函数
def read_file_lines(filename, from_size, line_num):
    i = 0
    text = []
    end_num = from_size + line_num 
    for line in open(filename):    #读取文件并将内容append进text列表
        if(i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text #返回列表


#加载数据和标签函数
def load_data_and_labels(filepath, max_size = -1):
    """
    Loads MR polarity data from files,  splits the data into words and generates labels.
    Returns split sentences and labels.
    从文件加载MR极性数据，将数据拆分为单词并生成标签。
    返回拆分语句和标签。
    """
    # Load data from files
    train_datas = []

    with open(filepath,  'r',  encoding='utf-8', errors='ignore') as f:
        train_datas = f.readlines()

    one_hot_labels = []  #热门标签
    x_datas = []
    for line in train_datas:
        line = line.strip()
        parts = line.split('\t', 1) #以“\t”分割，分割1次
        if(len(parts[1].strip()) == 0):
            continue

        words = jieba.cut(parts[1])  #分词
        x_datas.append(' '.join(words)) #将结果写进x_data列表
        
        one_hot = [0, 0, 0, 0]#垃圾共4类
        one_hot[int(parts[0])] = 1
        one_hot_labels.append(one_hot)
        
    print (' data size = ' , len(train_datas))

    # Split by words
    # x_text = [clean_str(sent) for sent in x_text]

    return [x_datas,  np.array(one_hot_labels)] #返回分词结果和热门标签的矩阵表示


#批处理函数
def batch_iter(data,  batch_size,  num_epochs,  shuffle=True):
    """
    Generates a batch iterator for a dataset.
    为数据集生成批处理迭代器。
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        #对每个过程洗数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))#对列表进行随机排序
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)

            # print('epoch = %d, batch_num = %d, start = %d, end_idx = %d' % (epoch, batch_num, start_index, end_index))
            yield shuffled_data[start_index:end_index]

#索引值获取函数
def get_text_idx(text, vocab, max_document_length):
    text_array = np.zeros([len(text),  max_document_length], dtype=np.int32)

    for i, x in  enumerate(text):
        words = x.split(" ")
        for j,  w in enumerate(words):
            if w in vocab:
                text_array[i,  j] = vocab[w]
            else :
                text_array[i,  j] = vocab['unknown']

    return text_array

#主函数
#加载文件内容并调用对内容处理的函数
if __name__ == "__main__":
    x_text,  y = load_data_and_labels('./data/data.txt')
    print (len(x_text))

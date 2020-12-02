# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import #绝对引入
from __future__ import division        #精确除法，本来的3/4=0 ==> 3/4=0.75
from __future__ import print_function  #即使在python2.X，使用print就得像python3.X那样加括号使用。

import argparse #argparse也就是一个方便用户添加命令行的库
import os.path
import re
import sys
import tarfile #主要作用是用来加压缩和解压缩文件

import numpy as np
from six.moves import urllib
import tensorflow as tf
#import tensorflow.compat.v1 as tf
FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels. 
  将整数节点ID转换为人类可读的标签。"""
#导入文件
  def __init__(self, 
                uid_chinese_lookup_path, 
                model_dir, 
                label_lookup_path=None,
                uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    #self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
    self.node_lookup = self.load_chinese_map(uid_chinese_lookup_path)
    

  def load(self, label_lookup_path, uid_lookup_path):
    """为每个softmax节点加载可读的英文名称。
    参数：
    label_lookup_path：字符串UID到整数节点ID。
    uid_lookup_path：字符串uid到人类可读字符串。
    返回：
    从整数节点ID到人类可读字符串的dict。
    """
    if not tf.gfile.Exists(uid_lookup_path): 
    #判断目录或文件是否存在，filename可为目录路径或带文件名的路径，有该目录则返回True，否则False
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # #加载从字符串UID到人类可读字符串
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()#tf.gfile.GFile类似于python提供的文本操作open()函数
    uid_to_human = {}
    #p = re.compile(r'[n\d]*[ \S,]*')
    #利用正则表达式提取需要的信息
    p = re.compile(r'(n\d*)\t(.*)')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      print(parsed_items)
      uid = parsed_items[0]
      human_string = parsed_items[1]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    #加载从字符串UID到整数节点ID的映射
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'): #startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是返回 True，否则False
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    #加载整数节点ID到人类可读字符串的最终映射
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name
    
  def load_chinese_map(self, uid_chinese_lookup_path):
    # Loads mapping from string UID to human-readable string
    #将UID从字符串加载到可读字符串
    proto_as_ascii_lines = tf.gfile.GFile(uid_chinese_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'(\d*)\t(.*)')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      #print(parsed_items)
      uid = parsed_items[0][0]
      human_string = parsed_items[0][1]
      uid_to_human[int(uid)] = human_string
    
    return uid_to_human

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph(model_dir):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  #从保存的GraphDef文件创建图形并返回一个保存程序。
  #从保存的图形创建图形_定义pb.
  #tf.gfile.FastGFile与tf.gfile.GFile的差别不大，速度快些
  with tf.gfile.FastGFile(os.path.join( 
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#图像的推测
def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.从保存的GraphDef创建图形。
  create_graph(FLAGS.model_dir)
  
   #创建会话
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across包含规范化预测的张量
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.通过将图像数据作为输入到图形来运行softmax
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # 创建节点标识-->中文字符串查找。
    node_lookup = NodeLookup(uid_chinese_lookup_path='./data/imagenet_2012_challenge_label_chinese_map.pbtxt', \
                                model_dir=FLAGS.model_dir)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      #print('node_id: %s' %(node_id))

#下载文件
def maybe_download_and_extract():
  """Download and extract model tar file."""
  #下载并提取模型tar文件。
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  
  
  #调用 add_argument() 方法添加参数，参数第一个是选项，第二个是数据类型，第三个不设置是默认值，第四个是help命令时的说明
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  #parser.parse_known_args()
  #有时间一个脚本只需要解析所有命令行参数中的一小部分，剩下的命令行参数给两一个脚本或者程序。在这种情况下，parse_known_args()就很有用。
  #它很像parse_args()，但是它在接受到多余的命令行参数时不报错。
  FLAGS, unparsed = parser.parse_known_args() 
  
  #tf.app.run()处理flag解析，然后执行main函数
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

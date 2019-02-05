import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = r'./tensorboard'

from collections import namedtuple
Label = namedtuple('Label','key mode person device')

def tensorview(vec_data,label_data):
    df=pd.DataFrame.from_records(data=vec_data)
    tf_data = tf.Variable(df.values.transpose())
    with tf.Session as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'tf_data.ckpt'))
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name

        # Link this tensor to its metadata(Labels) file
        metadata = os.path.join(LOG_DIR, 'metadata.tsv')
        with open(metadata, 'w+',encoding='utf8') as metadata_file:
            metadata_file.write('\t'.join(Label._fields))
            metadata_file.write('\n')
            for key in label_data.keys():
                metadata_file.write('\t'.join(label_data[key]))
                metadata_file.write('\n')
        embedding.metadata_path = 'metadata.tsv'
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
        print(f'create logs on {LOG_DIR}')

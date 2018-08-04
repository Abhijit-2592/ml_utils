"""
Tensorboard plugins.

# @Author: abhijit
# @Date:   2018-08-04T08:30:07+05:30
# @Last modified by:   abhijit
# @Last modified time: 2018-08-04T09:47:32+05:30
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import matplotlib.pyplot as plt


class TB_Projector(object):
    """Embedding projector wrapper class for tensorboard.

    Constructor Keyword Arguments:
    name --str: name of the embeddings
    variable --np.ndarray: a numpy array containing the embedding to be visualized
    labels --list: A list containing the corresponding labels for the variables
    log_dir --str: path to the logdir to save tensorboard summaries
    """

    def __init__(self, name, variable, labels, log_dir):
        """constructor."""
        assert isinstance(labels, list), "labeles must be a list of labels"
        assert len(labels) == len(variable), "The length of variables and length of labels must match"
        self.name = name
        self.variable = variable
        self.labels = labels
        self.log_dir = log_dir
        self.metadata_path = os.path.join(self.log_dir, "metadata.tsv")
        self.sprite_image_path = os.path.join(self.log_dir, "spriteimage.png")

    def project(self, sprite_image=None, single_sprite_image_dim=[]):
        """Embedding projector method.

        Keyword Arguments:

        """
        embedding_var = tf.Variable(self.variable, name=self.name)
        summary_writer = tf.summary.FileWriter(self.log_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = "metadata.tsv"
        with open(self.metadata_path, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(self.labels):
                f.write("%d\t%d\n" % (index, label))
        if sprite_image is not None:
            assert len(single_sprite_image_dim) == 2, "single_sprite_image_dim must be a list of 2 nums eg: [28,28] for mnist"
            embedding.sprite.image_path = "spriteimage.png"
            embedding.sprite.single_image_dim.extend(single_sprite_image_dim)
            plt.imsave(self.sprite_image_path, sprite_image, cmap='gray')

        projector.visualize_embeddings(summary_writer, config)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.log_dir, "model.ckpt"), 1)

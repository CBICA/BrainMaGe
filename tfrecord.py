import os
import glob

import nibabel as nib
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=[value]))

def tfrecordreader(filename_queue, dtype):
    options = tf.python_io.TFRecordOptions( tf.python_io.TFRecordCompressionType.GZIP )
    reader = tf.TFRecordReader(options = options)
    _, serialized_example = reader.read( filename_queue )
    feature = { 'image': tf.FixedLenFeature( [], tf.string ),
                'label': tf.FixedLenFeature( [], tf.string ) }
    features = tf.parse_single_example( serialized_example, features=feature )
    image = tf.decode_raw( features['image'], tf.dtypes.as_dtype(dtype))
    label = tf.decode_raw( features['label'], tf.int8 )
    image = tf.reshape(image, list( [128, 128, 128, 1] ))
    label = tf.reshape(label, list( [128, 128, 128, 1] ))
    return image, label

def tfrecordwriter(record_name, image, label, dtype):
    options = tf.python_io.TFRecordOptions( tf.python_io.TFRecordCompressionType.GZIP )
    writer = tf.python_io.TFRecordWriter( record_name, options=options )
    feature = { 'label': _bytes_feature( tf.compat.as_bytes( label.tostring() ) ),
                'image': _bytes_feature( tf.compat.as_bytes( image.tostring() ) ) }
    example = tf.train.Example( features=tf.train.Features(feature=feature) )
    writer.write( example.SerializeToString() )
    writer.close()
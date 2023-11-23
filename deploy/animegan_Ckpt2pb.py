# -*- coding: utf-8 -*-
import os

# tensorflow2.x
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

# tensorflow1.x
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(model_folder, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB save dir
    :return:
    '''
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # input node and output node from the network ( AnimeGANv2 generator)
    # input_op = 'generator_input:0'
    # output_op = 'generator/G_MODEL/out_layer/Tanh:0'

    output_node_names = "generator_1/main/out_layer"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,  # :sess.graph_def
            output_node_names=output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        for op in graph.get_operations():
            print(op.name, op.values())

if __name__ == '__main__':
    model_folder = "../checkpoint/generator_v3_Hayao_weight"
    pb_save_path = "AnimeGANv3_Hayao_36.pb"
    freeze_graph(model_folder, pb_save_path)


    """ pb model 2 onnx command"""
    cmd = f"python -m tf2onnx.convert --input {pb_save_path} --inputs AnimeGANv3_input:0  --outputs generator_1/main/out_layer:0  --output {pb_save_path[:-3]}.onnx"
    res = os.system(cmd)
    print(res)

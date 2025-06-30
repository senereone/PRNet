import tensorflow as tf


def freeze_graph_name(input_checkpoint):
    """
    :param input_checkpoint:
    :return:
    """
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=[var.name[:-2] for var in tf.global_variables()])
        for op in graph.get_operations():
            print(op.name, ": ", op.values())


if __name__ == '__main__':
    input_checkpoint = './model2/model.ckpt-330'
    freeze_graph_name(input_checkpoint)

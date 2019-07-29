# coding:utf-8
# Date : 2019.07.23
# author: phecda
#
#
# ********************************

import tensorflow as tf
from utils.utility import genHeadInfo

class KwsModel():
    def __init__(self):
        self.model_index = 'a'


    def load_labels(self, filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.gfile.GFile(filename)]

    def load_graph(self, filename):
        """Unpersists graph from file as default graph."""
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name=self.model_index)

    def run_graph(self, wav_data, labels, input_layer_name, output_layer_name,
                  num_top_predictions):
        """Runs the audio data through the graph and prints predictions."""
        with tf.Session() as sess:
            # Feed the audio data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

            # Sort to show labels in order of confidence
            top_k = predictions.argsort()[-num_top_predictions:][::-1]
            result_list = []
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                result_list.append((human_string, score))
                # print('%s (score = %.5f)' % (human_string, score))
            return result_list

    def label_wav(self, wav, labels, graph, how_many_labels, input_name=None, output_name=None):
        """Loads the model and labels, and runs the inference to print predictions."""
        if not wav or not tf.gfile.Exists(wav):
            tf.logging.fatal('Audio file does not exist %s', wav)

        if not labels or not tf.gfile.Exists(labels):
            tf.logging.fatal('Labels file does not exist %s', labels)

        if not graph or not tf.gfile.Exists(graph):
            tf.logging.fatal('Graph file does not exist %s', graph)

        labels_list = self.load_labels(labels)

        # load graph, which is stored in the default session
        self.load_graph(graph)

        # input output
        input_name = str(self.model_index) + "/wav_data:0"
        output_name = str(self.model_index) + "/labels_softmax:0"

        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()

        result_list = self.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
        return result_list

    def label_stream(self, buffer, labels, graph, how_many_labels, input_name=None, output_name=None):
        """Loads the model and labels, and runs the inference to print predictions."""
        if not labels or not tf.gfile.Exists(labels):
            tf.logging.fatal('Labels file does not exist %s', labels)

        if not graph or not tf.gfile.Exists(graph):
            tf.logging.fatal('Graph file does not exist %s', graph)

        labels_list = self.load_labels(labels)

        # load graph, which is stored in the default session
        self.load_graph(graph)

        # input output
        input_name = str(self.model_index) + "/wav_data:0"
        output_name = str(self.model_index) + "/labels_softmax:0"

        wav_data = genHeadInfo(16000, 16, len(buffer),1) + buffer

        result_list = self.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
        return result_list



if __name__ == "__main__":
    kws = KwsModel()
    re = kws.label_wav('../silence.wav', '../Pretrained_models/labels.txt', '../Pretrained_models/DS_CNN/DS_CNN_S.pb',3)
    print(re)
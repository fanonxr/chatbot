import time

import numpy as np
import tensorflow as tf

import config

# model implementation based on
# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/assignments/chatbot/model.py

class ChatBot:
    def __init__(self, forward_only, batch_size):
        """
        forward_only: if set, we do not construct the backward pass in the model.
        """

        print('Initialize new model')
        self.forward_only = forward_only
        self.batch_size = batch_size


    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i)) for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i)) for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i)) for i in range(config.BUCKETS[-1][1] + 1)]

        # the decoder inputs - shifted by 1 ignore the <GO> symbol
        self.targets = self.decoder_inputs[1:]


    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            weight = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            bias = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (weight, bias)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(weight),
                                                biases=bias,
                                                inputs=logits,
                                                labels=labels,
                                                num_sampled=config.NUM_SAMPLES,
                                                num_classes=config.DEC_VOCAB)

        self.softmax_loss_function = sampled_loss

        # single_cell = tf.contrib.rnn.LSTMBlockCell(config.HIDDEN_SIZE)
        single_cell = tf.contrib.rnn.GRUBlockCellV2(config.HIDDEN_SIZE)  # V2 - better for CPU
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.NUM_LAYERS)])


    def _create_loss(self):
        print("Creating loss")

        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # setattr(tf.contrib.rnn.LSTMBlockCell, '__deepcopy__', lambda self, _: self)
            # setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.GRUBlockCellV2, '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols=config.DEC_VOCAB,
                    embedding_size=config.HIDDEN_SIZE,
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

        if self.forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmax_loss_function
                                        )

            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output,
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]
                                            ]

        else:

            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function
                                        )

        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer')

        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.forward_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LEARNING_RATE)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []

                start = time.time()
                for bucket in range(len(config.BUCKETS)):

                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                trainables),
                                                                config.MAX_GRAD_NORM
                                                                )

                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                            global_step=self.global_step)
                                                            )

                    print(f'Creating opt for bucket {bucket} took {time.time()} seconds')
                    start = time.time()


    def build_graph(self):
        print("Building graph")
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
import os

import tensorflow as tf

import config

from encodermodel import Encoder
from bahdanauAttention import BahdanauAttention
from decodermodel import Decoder

# create encoder and decoder
encoder = Encoder(config.NUM_SAMPLES, config.BUCKETS, config.NUM_LAYERS, config.BATCH_SIZE)
decoder = Decoder(10, config.BUCKETS, config.NUM_LAYERS, config.BATCH_SIZE)

# create optimizer and loss
optimizer = tf.keras.optimizers.Adamax()
loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder,
                                decoder=decoder)


#@tf.function
# def train_step(inp, targ, enc_hidden):
#     loss = 0

#    with tf.GradientTape() as tape:
#         enc_output, enc_hidden = encoder(inp, enc_hidden)

#        dec_hidden = enc_hidden

#       dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * config.BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
#        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
#            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

#            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
#            dec_input = tf.expand_dims(targ[:, t], 1)

#    batch_loss = (loss / int(targ.shape[1]))

#    variables = encoder.trainable_variables + decoder.trainable_variables

#    gradients = tape.gradient(loss, variables)

#   optimizer.apply_gradients(zip(gradients, variables))

#    return batch_loss
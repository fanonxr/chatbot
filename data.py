import os
import random
import re

import numpy as np

import config

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def get_lines():
    """
    Getting seperate lines from movies.txt file
    Returns: a dict of an id and value = the sentence
    """
    # will be stored in a dict
    id2line = {}

    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE) # movies.txt file
    print(config.LINE_FILE)

    with open(file_path, 'r', errors='ignore') as file:

        i = 0
        try:
            for line in file:
                # split on
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        # will only get the actual sentenace/phrase
                        parts[4] = parts[4][:-1]

                    # store it in the dict
                    id2line[parts[0]] = parts[4]

                # increment the id counter
                i += 1

        except UnicodeDecodeError:
            print(i, line)

    return id2line

def get_convos():
    """
    Get conversations from the raw data
    returns a list of tags and conversation IDs
    """
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    # store the convos in a list
    convos = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                # only getting the convo ids
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])

                convos.append(convo)

    return convos

def question_answers(id2line, convos):
    """
        Divide the dataset into two sets:
        questions and answers.
        returns a seperate list of questions and answers
    """
    questions = []
    answers = []

    for convo in convos:
        for index, line in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]]) # adding the question to the list
            answers.append(id2line[convo[index + 1]])  # adding the answer - of that question to the list

    # debugging - make sure they are the same length
    assert len(questions) == len(answers)

    return questions, answers

def prepare_dataset(questions, answers):
    """Creating a file that will hold the train & test files of the encoder and decoder"""

    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], config.TESTSET_SIZE)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []

    # write the data to each specific test and train file
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename),'w'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i] + '\n')
            files[3].write(answers[i] + '\n')
        else:
            files[0].write(questions[i] + '\n')
            files[1].write(answers[i] + '\n')

    for file in files:
        file.close()


def tokenizer(line, normalize_digits=True):
    """
    Simple tokenizer
    clean the line with regex
    split into tokens for model to be able to process
    """
    # take out unneeded info on the line
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)

    # store the tokens in a list
    words = []

    # create our pattern objects to split on
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d") # will sub any digits for #


    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)

            words.append(token)

    return words

def build_vocab(filename, normalize_digits=True):
    """
    Building a vocabulary that the chatbot will utilize for training + testing
    these are the only words the chat bot can use to speak
    """
    # get paths
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, f'vocab.{filename[-3:]}')

    # a dict of
    vocab = {}

    with open(in_path, 'r') as file:
        for line in file.readlines():
            for token in tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0 # add the token to the dict if not already in
                vocab[token] += 1 #

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    with open(out_path, 'w') as file:
        # add PAD, UNK, START, and STOP tokens to the vocab
        file.write('<pad>' + '\n')
        file.write('<unk>' + '\n')
        file.write('<s>' + '\n')
        file.write('<\s>' + '\n')

        index = 4

        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            # write the word to the vocab file
            file.write(word + '\n')
            index += 1

        with open('config.py', 'a') as config_file:
            if filename[-3:] == 'enc':
                config_file.write('ENC_VOCAB = ' + str(index) + '\n')
            else:
                config_file.write('DEC_VOCAB = ' + str(index) + '\n')

def load_vocab(vocab_path):
    """Loading vocab
    returning a list of words and a dict of a word with a id
    """
    with open(vocab_path, 'r') as file:
        # create a list of words
        words = file.read().splitlines()

    # return a list of words and a dict
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    """Give a line a id """
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(line)]

def token2id(data, mode):
    """
    Convert all the tokens in the data into their corresponding
    index in the vocabulary.
    giving each token an id - makes data retrievl faster
    """

    # getting paths
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    words, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')

    # convert to list of lines
    lines = in_file.read().splitlines()

    for line in lines:
        if mode == 'dec': # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []

        # The length of the list increases by number of elements in it’s argument.
        ids.extend(sentence2id(vocab, line))


        if mode == 'dec':
            ids.append(vocab['<\s>'])

        # write to token to id
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data():
    print('Preparing raw data into train set and test set')

    print("Getting lines from movie_lines..")
    id2line = get_lines()

    print("Getting conversations from Movie_conversations")
    convos = get_convos()

    print("Creating a list of questions and a list of answers")
    questions, answers = question_answers(id2line, convos)

    print("Prepaing data set")
    prepare_dataset(questions, answers)

def process_data():
    print('Preparing data to be model-ready ...')

    print("building vocab for the encoder")
    build_vocab('train.enc')

    print("Building vocab for the decoder")
    build_vocab('train.dec')

    print("Tokenizing encoder.train")
    token2id('train', 'enc')

    print("Tokenizing decoder.train")
    token2id('train', 'dec')

    print("Tokenizing decoder.test")
    token2id('test', 'enc')

    print("Tokenizing decoder.test")
    token2id('test', 'dec')

def load_data(enc_filename, dec_filename, max_training_size=None):
    """
    Loading in our data
    """
    # get file paths

    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'r')

    encode, decode = encode_file.readline(), decode_file.readline()

    # bucketing we make different bucket for some max_len and we do this to reduce the amount of padding,
    # after making different buckets we train different model on different bucket.
    data_buckets = [[] for _ in config.BUCKETS]

    i = 0

    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)

        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]

        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break

        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    # a list of encoder and decoder ids
    return data_buckets

def _pad_input(input_, size):
    """
    adding padding to user input
    """
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """
    Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []

    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id] for batch_id in range(batch_size)], dtype=np.int32))

    #for i in batch_inputs:
    #   print(i)

    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket

    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)

        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []

    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)

        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0

        batch_masks.append(batch_mask)

    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

if __name__ == '__main__':
    prepare_raw_data()

process_data()
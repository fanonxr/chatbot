# parameters for processing the dataset
DATA_PATH = 'cornell-movie-dialogs-corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]

CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

# hyper parameters for the model

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LEARNING_RATE = 0.03 # 0.5 0.8 # 0.03
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512

# encoder and decoder vocabs

ENC_VOCAB = 24341
DEC_VOCAB = 24571
ENC_VOCAB = 24341
DEC_VOCAB = 24571
ENC_VOCAB = 24341
DEC_VOCAB = 24571
ENC_VOCAB = 24341
DEC_VOCAB = 24571
ENC_VOCAB = 20593
DEC_VOCAB = 20756
ENC_VOCAB = 20593
DEC_VOCAB = 20756
ENC_VOCAB = 20593
DEC_VOCAB = 20756
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281
ENC_VOCAB = 20187
DEC_VOCAB = 20281

# Data & Package Directory
# Paths are relative to `code/` (cwd when running scripts). Sibling repos live under
# F:\编程\DJtransGAN\{dg-pipeline,results}.
PAIR_DIR    = '../dg-pipeline/data/track/pair'
MIX_DIR     = '../dg-pipeline/data/mix'
STORE_DIR   = '../results'


# STFT Parameters
WINDOW      = 'hann'
N_FFT       = 2048  # 256 lose time resolution
HOP_LENGTH  = 512   # 128
N_MELS      = 128

# Band Parameters
BAND_FREQS = [20, 300, 5000, 20000]


# Others
SR          = 44100 # sampling rate
EPSILON     = 1e-12 # avoid divide zeros
RANDOM_SEED = 0     # random seed
N_TIME      = 60 
CUE_BAR     = 8
import sys
import os
import requests
import numpy as np
import tempfile
import pickle

cmu_url = sys.argv[1]
directory = sys.argv[2]

words = []
words_phonemes = []
with tempfile.NamedTemporaryFile() as write_file:
    r = requests.get(cmu_url, stream=True)
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            write_file.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    write_file.flush()
    with open(write_file.name, 'r') as read_file:
        for line in read_file:
            if not line.startswith(";;;"):
                parts = line.split()
                words.append(parts[0])
                words_phonemes.append(parts[1:])

unique_chars = set([ch for word in words for ch in word])
char2id = {}
id2char = {}
for i, ch in enumerate(unique_chars):
    char2id[ch] = i
    id2char[i] = ch

words_len = np.array([len(word) for word in words], dtype=np.int32)
max_words_len = np.amax(words_len)
words = [[char2id[ch] for ch in word] for word in words]
words = np.array([np.pad(np.array(word, dtype=np.int32), (0, max_words_len - len(word)), 'constant', constant_values=(0, 0)) for word in words], dtype=np.int32)

unique_phonemes = set([phone for phonemes in words_phonemes for phone in phonemes])
unique_phonemes.add('<eos>')
phoneme2id = {}
id2phoneme = {}
for i, phone in enumerate(unique_phonemes):
    phoneme2id[phone] = i
    id2phoneme[i] = phone

words_phonemes = [phonemes + ['<eos>'] for phonemes in words_phonemes]
words_phonemes_len = np.array([len(phonemes) for phonemes in words_phonemes], dtype=np.int32)
max_words_phonemes_len = np.amax(words_phonemes_len)
words_phonemes = [[phoneme2id[phonem] for phonem in phonemes] for phonemes in words_phonemes]
words_phonemes = np.array([np.pad(np.array(phonemes, dtype=np.int32), (0, max_words_phonemes_len - len(phonemes)), 'constant', constant_values=(0, 0)) for phonemes in words_phonemes], dtype=np.int32)

np.savez(os.path.join(directory, 'cmu_data'), X=words, X_seq_len=words_len, Y=words_phonemes, Y_seq_len=words_phonemes_len)
with open(os.path.join(directory, 'cmu.pkl'), 'wb') as meta_file:
    pickle.dump({
        "char2id": char2id,
        "id2char": id2char,
        "phoneme2id": phoneme2id,
        "id2phoneme": id2phoneme
    }, meta_file)

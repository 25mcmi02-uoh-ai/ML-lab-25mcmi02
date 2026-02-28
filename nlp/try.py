import re

# Byte Pair Encoding (BPE)

def byte_pair_encoding(corpus, limit=100):
    # get initial vocabulary
    # replace space with '_' then add space after every character in the corpus

    corpus = corpus.replace(" ", "_")
    vocabulary = sorted(list(set(list(corpus))))
    
    corpus = list(corpus)
        
    # repeat k times
    for i in range(limit):
        # find the most frequent pair
        frequent_pair = get_most_frequent_pair(corpus, vocabulary)
        
        if len(frequent_pair) == 0:
            break
        # add to the vocabulary
        vocabulary.append(frequent_pair[0]+frequent_pair[1])
        
        # modify the corpus
        replace_pair(corpus, frequent_pair)
        
    return vocabulary

def get_most_frequent_pair(corpus, vocabulary):
    most_frequent = []
    most_frequency = 0

    for v1 in vocabulary:
        for v2 in vocabulary:
            if v1[-1] == '_':
                break
            current = find_count_pair(corpus, v1, v2)
            if current > most_frequency:
                most_frequent = [v1, v2]
                most_frequency = current

    return most_frequent

def find_count_pair(corpus, v1, v2):
    counter = 0
    s = 1
    l = len(corpus)

    while True:
        try:
            found = corpus.index(v1, s)
        except ValueError:
            break
        if found+1 != l and corpus[found+1] == v2:
            counter += 1
        s = found+1
    return counter

def replace_pair(corpus, frequent_pair):
    s = 0
    length = len(corpus)
    v1 = frequent_pair[0]
    v2 = frequent_pair[1]

    while True:
        try:
            found = corpus.index(v1, s)
        except ValueError:
            break
        if found+1 < length:
            if corpus[found+1] == v2:
                # drop both elements from list and add the combined in that location
                corpus.insert(found, v1+v2)
                corpus.pop(found+1)
                corpus.pop(found+1)
        else:
            s = found+1


corpus = "low low low low lower lower lower new new new new new newer newer newer"
print(byte_pair_encoding(corpus, limit=10))
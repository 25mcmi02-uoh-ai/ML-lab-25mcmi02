# Byte Pair Encoding (BPE)

def byte_pair_encoding(corpus, limit=100):
    # get initial vocabulary
    vocabulary = sorted(list(set(list(corpus))))
    
    # replace space with '_' then add space after every character in the corpus
    corpus = corpus.replace(" ", "_")
    corpus = list(corpus)
        
    # repeat k times
    for i in range(limit):
        # find the most frequent pair
        frequent_pair = get_most_frequent_pair(corpus, vocabulary)
        
        # add to the vocabulary
        vocabulary.append(frequent_pair[0]+frequent_pair[1])
        
        # modify the corpus
        replace_pair(corpus, frequent_pair)
        
    return vocabulary

def get_most_frequent_pair(corpus, vocabulary):
    


    return ['e', 'r']

def replace_pair(corpus, frequent_pair):
    length = len(corpus)
    v1 = frequent_pair[0]
    v2 = frequent_pair[1]
    
    while True:
        try:
            found = corpus.index(v1)
        except ValueError:
            break

        if corpus[found+1] == v2:
            # drop both elements from list and add the combined in that location
            corpus.insert(found, v1+v2)
            corpus.pop(found+1)
            corpus.pop(found+2)


corpus = "low low low low lower lower lower new new new new new newer newer newer"
print(byte_pair_encoding(corpus, limit=3))
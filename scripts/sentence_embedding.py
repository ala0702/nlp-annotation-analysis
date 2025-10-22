from gensim.models.fasttext import load_facebook_vectors
from flair.data import Sentence

sentence = Sentence("Myśl z duszy leci bystro, Nim się w słowach złamie.")

embedding = load_facebook_vectors("../models/kgr10.plain.cbow.dim100.neg10.bin")
embedding.embed([sentence])

for token in sentence:
    print(token)
    print(token.embedding)

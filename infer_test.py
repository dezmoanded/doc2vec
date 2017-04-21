#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs

#parameters
model="/Users/paul/Documents/kaggle/quora/enwiki_dbow/doc2vec.bin"
test_docs = "/Users/paul/Documents/kaggle/quora/question1.txt"
output_file = "/Users/paul/Documents/kaggle/quora/question1_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#infer test vectors
output = open(output_file, "w")
i = 0
for d in test_docs:
    output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
    if i % 500 == 0:
        print str(i) + " / " + str(len(test_docs)) + "\t\t" + str(d)[:40] + "\n"
    i += 1
output.flush()
output.close()

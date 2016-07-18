import nltk
import os
import pandas as pd
from edit_distance import stringdist
import multiprocessing as mp
from sentence_lib import get_word_from_document
import mysql.connector
document_queue = mp.Queue()



def read_from_file(args):
    file_name = args
    # print(file_name)
    # print
    open_file = open('../Data/'+file_name,'r')
    data = open_file.read()
    open_file.close()
    sentences,list_of_words = get_word_from_document(data)

    document_queue.put((args,sentences,list_of_words))


def main():

        print(os.getpid())
        list_name_of_file = []
        for i in range(2):
            if i%2 == 0:
                list_name_of_file+=['pg25990.txt']
            else:
                list_name_of_file+=['pg706.txt']

        # list_name_of_file = ['pg25990.txt','pg706.txt']
        # read_from_file(list_name_of_file[0])
        print(len(list_name_of_file))
        pool = mp.Pool(mp.cpu_count())
        pool.map(read_from_file,list_name_of_file)
        list_of_document = []
        z = 0
        for i in range(len(list_name_of_file)):
            list_of_document.append(document_queue.get())

            z += len(list_of_document[i][2])
            # list_of_words = uniq_word(list/_of_words)
        # print(list_of_document[0][0])
        print(z)
    # path_of_file = '../Data/pg25990.txt'
    # sentences, list_of_sentences = get_word_to_list(path_of_file)
    # x = find_same_cluster(list_of_sentences[1],4)
    # print(len(sentences))
    # print ("SEARCH SENTENCE = "+sentences[1])
    # cos_dist = []
    # for i in range(len(list_of_sentences)):
    #     y = find_same_cluster(list_of_sentences[i],4)
    #     cos_dist.append((sentences[i],1-cosine_distance(x,y)))
    # x = find_same_cluster(,4)
    # print (x)

    # print (y)
    # print (len(intersection(x,y))/len(review_to_sentences(z,tokenizer,remove_stopwords=True)[0]))
    # print (intersection(x,y))

    # cos_dist = sorted(cos_dist,key = lambda x: x[1],reverse = True)
    # print_(cos_dist)



if __name__ == "__main__":

    main()

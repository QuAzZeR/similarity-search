import mysql.connector
from edit_distance import stringdist
import ast
def find_cluster(search_sentence,type_cluster = 3):
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost',
                                  database='similarity_search')
    cursor = cnx.cursor()
    # type_cluster = input("select type cluster DBSCAN == 3, KMEAN == 4 : ")
    quert = "Select * from cluster_of_word where book_id = "+str(type_cluster)
    # file_name = 'pg25990.txt'
    # f = open('../Data/'+file_name,'r')

    # book = {'name_of_book':file_name,'text':f.read()}
    cursor.execute(quert)
    # cnx.commit()
    # search_word = input('enter your word : ')
    temp = []
    for i in cursor:
        temp.append(i)
    cnx.close()
    ans_sentence = []
    for search_word in search_sentence:
        ans = []
        min = 1000
        for i in temp:

            if abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1])) < min:
                min = abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1]))
                ans = ast.literal_eval(i[5])
                # print (search_word,ans)
            elif abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1])) == min:
                ans += ast.literal_eval(i[5])
                # print (search_word,ans)

        ans_sentence.append(str(ans))
    return ans_sentence

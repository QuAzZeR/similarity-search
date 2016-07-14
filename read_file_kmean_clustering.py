import ast
def find_group(name_of_file):
    fi= open('../../KMEAN/'+name_of_file,'r+')
    f= fi.readlines()
    fi.close()
    ff = [i.replace('\n','') for i in f]
    # print (ff)
    # f = []
    z = []
    group =0
    Max = 0
    Min = 0
    list_of_word = []
    for i in ff:
        # pass
        # print(ff)
        if 'For n' in i:
            continue
        f= ast.literal_eval(i)
        if type(f) != tuple :
            continue
        z.append(f)
        if f[1][1] == 1.0:
            group = f[2]
    for i in z:
        if i[2] == group:
            pass
            # print(123)
    # print (group)
    # print(len(z))
    center_word = '.'
    for i in z:
        # print (i[2])
        if i[2] == group:
            list_of_word.append(i[0])
        if len(list_of_word) == 1:
            Max = i[1][1]
            Min = i[1][1]
        else:
            Max = max(Max,i[1][1])
            Min = min(Min,i[1][1])
        if i[1][1] == 1.0:
            count_f = 0
            count_n = 0
            count_same = 0
            for j in range(len(i[0])):
                if j> len(name_of_file):
                    break
                if name_of_file[j] =='.' or name_of_file[j]=='/' or name_of_file[j] == '_':
                    count_n += 1

                if i[0][j] == '.' or i[0][j] == '/':
                    count_f += 1
                if i[0][j] =='.' and i[0][j] == name_of_file[j]:
                    continue
                if i[0][j] == name_of_file[j] :
                    count_same += 1
            if count_same+count_f == len(name_of_file) :
                center_word = i[0]



            # if(name_of_file == i[0] or name_of_file.replace('_','/') == i[0] or name_of_file.replace('_','/') == i[0]):
                # center_word = i[0]


    return {'word' : str(list_of_word),'word_center' : center_word,'max' : Max, 'min' : Min}

Dict_word = {}
# Dict_word.append(find_group('oTHER'))
fi = open('name_of_file','r+')
fff = fi.readlines()
# print (fff)
for i in fff:
    Dict_word[i.replace('\n','')] = find_group(i.replace('\n',''))
print (Dict_word)


import mysql.connector

cnx = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='similarity_search')
cursor = cnx.cursor()

insert_cluster_of_book = "Insert into cluster_of_word(book_id,center_word,max_distance,min_distance,similarity_word) VALUE(4,%(word_center)s,%(max)s,%(min)s,%(word)s)"

for i in Dict_word:
    cursor.execute(insert_cluster_of_book,Dict_word[i])
    cnx.commit()
cnx.close()

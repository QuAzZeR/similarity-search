import math
def intersection(x,y):
    same = []
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                same.append(x[i])
                break
    return same

def union(x,y):
    same = []
    for i in range(len(x)):
        if not x[i] in same:
            same.append(x[i])
    for i in range(len(y)):
        if not y[i] in same:
            same.append(y[i])
    return same

def qgram(str1,q=1):
    return [str1[i:i+q] for i in range(0,len(str1)) if i+q <= len(str1)]

def qgram_distance(q1,q2):
    gxuniongy = (q1) + (q2)
    gxintersecgy = intersection(q1,q2)
    distance = len(gxuniongy)-2*len(gxintersecgy)
    norm_dist = (distance/(len(gxuniongy)-len(gxintersecgy)))
    return norm_dist

def cosine_distance(q1,q2):
    qgramI = len(intersection(q1,q2))*1.0
    # print q1,q2
    # print qgramI
    # print len(q1),len(q2)
    # print math.sqrt(len(q1)*len(q2))
    try:
        distance =  1 - qgramI/math.sqrt(len(q1)*len(q2))
        return distance
    except:
        return 1.0

def jaccard_distance(q1,q2):
    qgramU = len(union(q1,q2))*1.0
    qgramI = len(intersection(q1,q2))*1.0
    distance =  1 - qgramI/qgramU
    return distance

def ed_dynp(x,y):
    col0 = [i for i in range(len(x)+1)]
    # col1 = [0 for i in range(len(x)+1)]
    for i in range(len(x)+1):
        col0[i] = i
    # print(col0,col1)
    for j in range(1,len(y)+1):
        col1 = [0 for i in range(len(x)+1)]
        col1[0] = j

        for i in range(1,len(x)+1):
            if x[i-1] == y[j-1]:
                c = 0
            else:
                c = 1
            col1[i] = min(col0[i-1]+c,col1[i-1]+1,col0[i]+1)
            # print("=========================")
            # print(x[i-1],y[j-1],c)
            # print(col0)

            # print(col1)
            # print("=========")
        # print( col0)
        col0 = col1
        # print('---------------')
    return 1.0*col0[-1]/max(len(x),len(y))

def jw_distance(str1,str2,p):
    m = len(str1)*1.0
    n = len(str2)*1.0
    # print(intersection(str1,str2))
    # print ststr1)
    samefirst = []
    if len(str1) < len(str2):
        for i in range(len(str1)):
            if str1[i] == str2[i]:
                samefirst.append(str1[i])
            else:
                break

    else:
        for i in range(len(str2)):
            if str1[i] == str2[i]:
                samefirst.append(str2[i])
            else:
                break

    sametemp= intersection(str1,str2)
    same1 = [str1[i] for i in range(len(str1)) if str1[i] in sametemp]
    same2 = [str2[i] for i in range(len(str2)) if str2[i] in sametemp]

    # print (str1,str2)
    # print(sametemp,same1,same2,samefirst)
    c = len(sametemp)*1.0
    l = len(samefirst)*1.0

    if len(same1) < len(same2):
        temp = len([same1[i] for i in range(len(same1)) if same1[i]!=same2[i]])
    else:
        temp = len([same2[i] for i in range(len(same2)) if same1[i]!=same2[i]])
    t = (temp-temp%2)/2.0

    # print(c,m,n,t,l)
    try:
        dj = 1.0-1/3.0*(c/m+c/n+((c-t)/c))
        dw = dj*(1-l*p)
        return dw
    except:
        return "Infinity"
# def word2vec(word):
#     from collections import Counter
#     from math import sqrt
#
#     # count the characters in word
#     cw = Counter(word)
#     # precomputes a set of the different characters
#     sw = set(cw)
#     # precomputes the "length" of the word vector
#     lw = sqrt(sum(c*c for c in cw.values()))
#
#     # return a tuple
#     return cw, sw, lw
#
# def cosdis(str1, str2):
#     # which characters are common to the two words?
#     v1 = word2vec(str1)
#     v2 = word2vec(str2)
#
#     common = v1[1].intersection(v2[1])
#     distance = sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]
#     # if str1.lower() == str2.lower() and str1 != str2:
#     #     return distance+0.05
#     # by definition of cosine distance we have
#     return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]



def stringdist(str1,str2,method='cosine',q=1,p=0):
    q1 = qgram(str1,q)
    q2 = qgram(str2,q)
    # print(q1,q2)
    if(method == "jaccard"):
        return str2,jaccard_distance(q1,q2)
    elif(method == "cosine"):
        return str2,cosine_distance(q1,q2)
        # return str2,cosdis(str1,str2)
    elif(method == "qgram"):
        return str2,qgram_distance(q1,q2)
    elif(method =="ed"):
        return str2,ed_dynp(str1,str2)
    elif(method =="jw"):
        return str2,jw_distance(str1,str2,p)




# print(stringdist("Cosmo Kramer","Cosmo Kramer","jw",2,0.1))
# model = 'Cosmo Kramer'
# l = ['Cosmo Kramer','Cosmo X. Kramer','Comso Kramer','Coso Kraer','Csmo Kramer','Cosmer Kramo','Cosmer Kramo','Kosmo Kramer','Kosmoo Karme','C.o.s.m.o. .K.r.a.m.e.r','Sir Cosmo Kramer','Csokae','Mr. Kramer',' Ckaemmoorrs','remark omsoC','Kramer, Cosmo','George Costanza','Dr. Van Nostren','Jerry Seinfeld','Elaine Benes']
# f = open('./Log/edit_distance.csv','w+')
# f.write("WORD,Edit Distance,Jaccard,Qgram,Cosine,JARO\n")
#
# for i in range(len(l)):
#     f.write("\""+l[i]+"\""+","+ str(stringdist(model,l[i],"ed",2))+",")
#     f.write(str(stringdist(model.lower(),l[i].lower(),"jaccard",2))+",")
#     f.write(str(stringdist(model.lower(),l[i].lower(),"qgram",2))+",")
#     f.write(str(stringdist(model.lower(),l[i].lower(),"cosine",2))+",")
#     f.write(str(stringdist(model.lower(),l[i].lower(),"jw",2,0.1))+"\n")
# f.close()
# f = open('./Log/jaccard.log','w+')
# for i in range(len(l)):
#     f.write(l[i]+","+ str(stringdist(model,l[i],"jaccard",2))+"\n")
# f.close()
# f = open('./Log/qgram.log','w+')
# for i in range(len(l)):
#     f.write(l[i]+","+ str(stringdist(model,l[i],"qgram",2))+"\n")
# f.close()
# f = open('./Log/cosine.log','w+')
# for i in range(len(l)):
#     f.write(l[i]+","+ str(stringdist(model,l[i],"cosine",2))+"\n")
# f.close()

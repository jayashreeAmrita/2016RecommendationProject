from __future__ import division

import numpy as np
import csv
import psycopg2


from random import randint
from math import  sqrt



g_testidx_i=0
g_testidx_j=0


arr = []
cent=[]

# generate data matrix
def dataset(train_i,train_j,test_i,test_j):

     inp = open ("C:\\Users\AJESH\Desktop\ml-1m/ratings.dat","r")
     for line in inp.readlines():
        arr.append([])
        for i in line.split('::'):
            arr[-1].append(int(i))

     data = np.zeros(shape=(6040,3952), dtype=np.int)

     for i in arr:
         data[(i[0])-1][(i[1])-1]= int(i[2])
     ''' # normalize values around 0
     for i in data:
         s1= data[i,:]
         m_n=s1.mean()
         for j in i:
             j=j- m_n'''
     data1=np.zeros(shape=(train_j-train_i,3952), dtype=np.int)
     data2=np.zeros(shape=(test_j-test_i,3952), dtype=np.int)

     data1=data[train_i:train_j,:]
     data2=data[test_i:test_j,:]
     writecsv(data2,"test1.csv")

     return data1

def rms_dataset(ti,tj):
     inp = open ("C:\\Users\AJESH\Desktop\ml-1m/ratings.dat","r")
     for line in inp.readlines():
        arr.append([])
        for i in line.split('::'):
            arr[-1].append(int(i))

     data = np.zeros(shape=(6040,3952), dtype=np.int)

     for i in arr:
         data[(i[0])-1][(i[1])-1]= int(i[2])
     # normalize values around 0
     '''for i in data:
         s1= data[i,:]
         m_n=s1.mean()
         for j in i:
             j=j- m_n'''
     data1=np.zeros(shape=(tj-ti,3952), dtype=np.int)


     data1=data[ti:tj,:]


     return data1


def rms_cluster(ti,tj,k,itre):
    cl=(kmean(rms_dataset(ti,tj),k,itre))
    writecsv(cl,"rms_cluster_result.csv")



def writecsv(data_file,file_name):
  with open(file_name, 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data_file)




# check if clusters have converged
def has_converged( iterations,numb):
    MAX_ITERATIONS = numb

    if iterations >= MAX_ITERATIONS:
        return True
    else:
        return False

# k++ centroids
def randomize_centroids(data, oldcentroids, k):

    j=[]
    rand_val=randint(1,len(data)-1)
    idx=0
    val=0
    eq_sm=0.0
    eq_sq=0.0
    eq_sq_d=0.0
    minm=0.0
    min_sq=0.0
    sum1=0.0

    for i in range(0,len(data)):
        minm=i-rand_val
        min_sq=sqrt(pow(minm,2))
        sum1=sum1+min_sq


    for idx  in range (0,k):
        val=randint(1,len(data)-1)
        if(val== rand_val):
            while(val!=rand_val):
                val=randint(1,len(data)-1)


        eq_sm=pow(val-rand_val,2)
        eq_sq=sqrt(eq_sm)
        j.append(eq_sq/sum1)
    return j
# ecludiean function
def ecludiean (dataset,index1,index2,rlen,clen):
     sum=0.0
     sq=0.0

     for i in range(0,clen):
          sum=(sum+ (pow((dataset[index1][i] - dataset[index2][i]),2)))
     sq=sqrt(sum)
     return (round(sq,1))




# kmeans
def kmean(data,k,iter):
    row_len=len(data)
    col_len=len(data[0])

    iterations=0
    centroids=[]
    mean_cluster=[]
    cluster=[[] for x in xrange(k)]
    mean_arr=np.zeros(shape=(len(data),k), dtype=np.float)


    while(has_converged(iterations,iter)== False):
        if(iterations==0):
               j_indx=0
               i_indx=0
               centroids=randomize_centroids(data,centroids,k)

               for i in centroids:
                   j_indx=0
                   for j in range(0,row_len):
                        eclu_val= ecludiean(data,i,j_indx,row_len,col_len)

                        mean_arr[j_indx][i_indx]=eclu_val

                        j_indx+=1

                   i_indx+=1
               print centroids
               '''-------------------------------------------------------'''

               i_indx=0
               indx=0
               for i in mean_arr:
                  small=0.0
                  j_indx=0
                  small=i[0]
                  for j in  i :

                     if (small>i[j_indx]):
                        small=i[j_indx]
                        indx=j_indx

                     j_indx+=1

                  cluster[indx].append(i_indx)

                  i_indx+=1



        elif (iterations>0):

               new_centroid=[[] for x in xrange(k)]
               sum1=0
               k_indx=0
               j_indx=0
               mean_val=0.0
               for j in cluster:

                   for i in range(0,col_len):
                       sum1=0
                       k_indx=0
                       for ki in j:
                           sum1+= data[ki][i]
                           k_indx+=1
                       if (sum1==0):
                           mean_val=0
                       else:
                           mean_val=sum1/len(j)
                       new_centroid[j_indx].append(round(mean_val,1))
                   j_indx+=1
               print new_centroid
               #----------------------------------------------------------------#

               val_indx=0
               mean_arr=np.zeros(shape=(len(data),k), dtype=np.float)
               val=0.0
               val1=0.0
               sq_val=0.0
               i_indx=0
               for i in range(0,k):
                   for j in range(0,row_len):
                       for kk in range(0,col_len):
                           val=data[j][kk]-new_centroid[i][kk]
                           sq_val=sq_val+pow(val,2)
                       mean_arr[j][i]=sqrt(sq_val)
                       sq_val=0
               #print mean_arr


               cluster=[[] for x in xrange(k)]
               sm=0.0
               sm_indx=0
               for lenn in range (0,row_len):
                   sm=mean_arr[lenn][0]
                   for jlen in range (0,k):
                       if(mean_arr[lenn][jlen]<sm):
                           sm=mean_arr[lenn][jlen]
                           sm_indx=jlen
                   cluster[sm_indx].append(lenn)
                   sm=0.0
                   sm_indx=0

               print(cluster)
        iterations+=1
    writecsv(new_centroid,"centroid.csv")

    return cluster


def compute_cluster(tr_i,tr_j,te_i,te_j,clus_num,itert):

    cl=(kmean(dataset(tr_i,tr_j,te_i,te_j),clus_num,itert))
    writecsv(cl,"cluster_result.csv")

    for i in range(0,len(cl)):
        print "Cluster "+str(i+1)+":"
        print cl[i]
        print "Cluster "+str(i+1)+" Count:"+str(len(cl[i]))
        print "------------------------------------------------------------------"

    datanw=np.zeros(shape=(len(dataset(tr_i,tr_j,te_i,te_j)),3953), dtype=np.int)
    da=dataset(tr_i,tr_j,te_i,te_j)


    for i in range(0,len(da)):
        for j in range (0,3952):
            datanw[i][j]=da[i][j]


    cl_indx=0
    for i in range (0,len(datanw)):
        cl_indx=0
        for j in range (0,len(cl)):
            if (cl[j].__contains__(i)):
                cl_indx=j
        datanw[i][3952]=cl_indx



    writecsv(datanw,"train1.csv")
    arr_indx=np.zeros(shape=(1,2), dtype=np.int)
    arr_indx[0,0]=te_i
    arr_indx[0,1]=te_j
    writecsv(arr_indx,"test_index.csv")
    rms_cluster(tr_i,te_j,clus_num,itert)



def dbwrite_cluster():
    conn = psycopg2.connect(database="postgres", user="postgres", password="as", host="127.0.0.1", port="5432")
    cur = conn.cursor()

    with open('cluster_result.csv', 'rb') as f:
        reader = csv.reader(f)
        cluster = list(reader)


    indx=0
    jindx=0
    for i in cluster:
        for j in i:
            cur.execute("INSERT INTO cluster (id, nameid, cluster) VALUES (%s,%s,%s)", (jindx, j, indx))
            jindx+=1
        indx+=1

    conn.commit()
    conn.close()

def dbwrite_recommendation(idx,jdx):
    conn = psycopg2.connect(database="postgres", user="postgres", password="as", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    curs = conn.cursor()

    inp = open("C:\\Users\AJESH\Desktop\ml-1m/ratings.dat", "r")
    for line in inp.readlines():
        arr.append([])
        for i in line.split('::'):
            arr[-1].append(int(i))

    data = np.zeros(shape=(6040, 3952), dtype=np.int)

    for i in arr:
        data[(i[0]) - 1][(i[1]) - 1] = int(i[2])

    print data



    with open('cluster_result.csv', 'rb') as f:
        reader = csv.reader(f)
        cluster = list(reader)

    res= [[] for x in xrange(len(cluster))]
    recd_val=0
    cnt=0
    for k in range(0,len(cluster)):
        for i in range (0,3951):

            for j in range (idx,jdx):
                 if str(j) in cluster[k]:
                     if (data[j][i]!=0):
                           print ("-----------"+str(i)+"-------"+str(j)+"----------")

                           recd_val=recd_val+ int(data[i][j])
                           cnt+=1
                           print (str(recd_val) + "-----" + str(data[j][i]))
                           print ("*****************************")
                           print recd_val
                           print cnt
                           print ("*******************************")
                           print ("----------------------------")

            if ((recd_val!=0)&(cnt!=0)):
                res[k].append(recd_val/cnt)
            else:
                res[k].append(0)

            recd_val = 0
            cnt = 0



    print len(res[0])
    jkdx=0
    t=0

    for i in range (0,len(res)):
        jkdx = 0
        for j in res[i]:
            if (j>0):
                curs.execute("SELECT  link , name  from movie where idd="+str(jkdx)+"")
                rows = curs.fetchall()
                for row in rows:
                   lnk= row[0]
                   ns=row[1]
                   print lnk
                   print jkdx
                cur.execute("INSERT INTO reco (id, clus, link,nam) VALUES (%s,%s,%s,%s)", (t, i, lnk,ns))
                t+=1
            jkdx+=1
    print res[0]
    print res[1]
    print res[2]








    conn.commit()
    conn.close()



#compute_cluster

# compute_cluster(0,300,300,1000,3,10)
#dataset_write(0,500,550,600,"test2.csv")

# dbwrite_cluster(0,300)
dbwrite_recommendation(0,300)















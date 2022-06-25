import sqlite3
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import itertools
from sklearn.decomposition import PCA
#Connection to the DB
conn = sqlite3.connect('/event.db')

cursor = conn. cursor()
cursor. execute("SELECT name FROM sqlite_master WHERE type='table';") # [('event_number',), ('event',), ('drug',), ('extraction',)]
print("Tables : ",cursor. fetchall())
cursor. execute("SELECT name FROM PRAGMA_TABLE_INFO('drug');") # [('index',), ('id',), ('target',), ('enzyme',), ('pathway',), ('smile',), ('name',)]
print("drug : ",cursor. fetchall())
cursor. execute("SELECT name FROM PRAGMA_TABLE_INFO('event');") # [('index',), ('id',), ('target',), ('enzyme',), ('pathway',), ('smile',), ('name',)]
print("event : ",cursor. fetchall())
cursor. execute("SELECT COUNT(*) FROM event") # [('index',), ('id',), ('target',), ('enzyme',), ('pathway',), ('smile',), ('name',)]
print("event row COUNT: ",cursor. fetchall())

# close the DB connection 
conn.close() 

def save_f(name,mat):
  print(name)
  print(mat.shape)
  df = pd.DataFrame(mat)
  df.to_csv('/DDI/'+str(name)+'.csv', header=None, index=False)

def Jaccard(matrix):
  matrix = np.mat(matrix)
  numerator = matrix * matrix.T
  denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
  return numerator / denominator

def feature_vector(df, feature_name):
  all_feature = []
  drug_list = np.array(df[feature_name]).tolist()
  # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
  for i in drug_list:
    for each_feature in i.split('|'):
      if each_feature not in all_feature:
        all_feature.append(each_feature)  # obtain all the features
  feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
  df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
  for i in range(len(drug_list)):
    for each_feature in df[feature_name].iloc[i].split('|'):
      df_feature[each_feature].iloc[i] = 1

  sim_matrix = Jaccard(np.array(df_feature))
  sim_matrix1 = np.array(sim_matrix)
  pca = PCA(n_components=len(sim_matrix1))  # PCA dimension
  pca.fit(sim_matrix)
  sim_matrix = pca.transform(sim_matrix)

  return sim_matrix

conn = sqlite3.connect('/event.db')
df_drug = pd.read_sql('select * from drug;', conn)

feature_list = ['target', 'enzyme', 'pathway', 'smile']

print(df_drug[feature_list[0]][:2])
print(df_drug[:][:2])

drugs = df_drug[:]
drugs = np.array(np.vstack((df_drug['index'],df_drug['name'])))
drugs = drugs.T

for feature in feature_list:
  mat = feature_vector(df_drug, feature)
  save_f(feature+"_PCA", mat)

conn.close()

conn = sqlite3.connect('/event.db')
extraction = pd.read_sql('select * from extraction;', conn)
mechanism = extraction['mechanism']
action = extraction['action']
drugA = extraction['drugA']
drugB = extraction['drugB']
d_label = {}
d_event=[]
for i in range(len(mechanism)):
  d_event.append(mechanism[i]+" "+action[i])
count={}
for i in d_event:
  if i in count:
    count[i]+=1
  else:
    count[i]=1
list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
for i in range(len(list1)):
  d_label[list1[i][0]]=i 

DDI=[]
for i in range(len(d_event)):
  DDI.append(np.hstack((d_label[d_event[i]],drugA[i], drugB[i])))

mat_DDI = np.array(DDI)
key = drugs[:,1]
val = drugs[:,0]
dic = dict(zip(key,val))
postive1 = [dic[item] for item in mat_DDI[:,1]]
postive2 = [dic[item] for item in mat_DDI[:,2]]
full_pos = np.array(np.vstack((mat_DDI[:,0],postive1,postive2))).astype('int32')
full_pos = full_pos.T

df = pd.DataFrame(np.array(full_pos).tolist())
df.to_csv('/DDI/full_pos2.txt', header=None, index=None, sep=' ')

conn.close()

def make_neg_pairs(matrix):
  all_pos = np.array(matrix)
  s1=np.unique(all_pos[:,0])
  s2=np.unique(all_pos[:,1])
  s3=set(s1).union(s2)
  conncted_drug = sorted(s3)
  print("there are ", len(s3), " drugs have connaction out of 572")
  ss = [ii for ii in range(572) if not ii in s3]
  print("there are ", len(ss), " drugs without connaction out of 572")
  pairs_false = list() 
  pairs = list()
  comparing = all_pos
  print("start callcolate combinations ... ")
  for dr1,dr2 in itertools.combinations(conncted_drug,2):
    d1=np.array([dr1,dr2])
    d2=np.array([dr1,dr2])
    if dr1 == dr2: continue
    else: pairs.append((dr1,dr2))
  print("all pairs : ",len(pairs))
  for dr in tqdm(pairs, desc="pairs_false generating : "):
    d1=np.array([dr[0],dr[1]])
    d2=np.array([dr[1],dr[0]])
    if not (dr[0]==dr[1]): 
      if not ((d2 == comparing).all(axis=1).any() or (d1 == comparing).all(axis=1).any()):
        pairs_false.append([dr[0],dr[1]])
  base=[]
  base2=[]
  for o in tqdm(pairs_false, "all_neg generating : "):
    if (not any(o[0] in h for h in base)) or (not any(o[1] in h for h in base)):
      base.append(o)
    else:
      base2.append(o)
  if len(base) > len(conncted_drug) : 
    print("less base .... !")
  pairs_f1 = np.array(base2)
  np.random.shuffle(pairs_f1)
  all_neg = np.concatenate((base,pairs_f1[:len(all_pos)-len(base)]),axis=0)
  np.random.shuffle(all_neg)
  print("all_neg.shape : ", all_neg.shape, "all_pos.shape : ", all_pos.shape)
  df = pd.DataFrame(np.array(all_neg).tolist())
  df.to_csv('/DDI/all_neg2.txt', header=None, index=None, sep=' ')
  return all_neg

all_neg = make_neg_pairs(full_pos[:,1:])

for itm in tqdm(range(9)):
  if itm == 0: 
    full_pos = np.array(np.array(pd.read_csv("DDI/full_pos2.txt", header=None , sep=' ')).tolist())
  elif itm == 1:
    all_neg = np.array(np.array(pd.read_csv("/DDI/all_neg2.txt", header=None , sep=' ')).tolist())
  elif itm == 2:
    target = np.array(np.array(pd.read_csv("/DDI/target_PCA.csv", header=None)).tolist())
    enzyme = np.array(np.array(pd.read_csv("/DDI/enzyme_PCA.csv", header=None)).tolist())
    pathway = np.array(np.array(pd.read_csv("/DDI/pathway_PCA.csv", header=None)).tolist())
    smile = np.array(np.array(pd.read_csv("/DDI/smile_PCA.csv", header=None)).tolist())
  elif itm == 3:
    full_pos = np.array(np.vstack((full_pos[:,0],full_pos[:,1],full_pos[:,2],[1]*len(full_pos)))).astype('int32').T
    all_cat_pos = []
    for i in range(65):
      all_cat_pos.append(([np.array(item).tolist() for item in full_pos if item[0]==i]))
  elif itm == 4:
    l_l = len(all_cat_pos[0])
    f_l = 0
    all_cat_neg = []
    for i in range(65):
      all_cat_neg.append(np.vstack(([i]* len(all_cat_pos[i]),all_neg[f_l:l_l,0].tolist(),all_neg[f_l:l_l,1].tolist(),[0]* len(all_cat_pos[i]))).T)
      f_l = l_l
      if i<64:
        l_l += len(all_cat_pos[i+1])
  elif itm == 5: 
    train_cat_pos = []
    train_cat_neg = []
    valid_cat_pos = []
    valid_cat_neg = []
    test_cat_pos = []
    test_cat_neg = []
    for i in range(65):
      train_cat_pos.append((all_cat_pos[i][:(len(all_cat_pos[i])*65//100)]))
      train_cat_neg.append((all_cat_neg[i][:(len(all_cat_neg[i])*65//100)]))
      valid_cat_pos.append((all_cat_pos[i][(len(all_cat_pos[i])*65//100):(len(all_cat_pos[i])*80//100)]))
      valid_cat_neg.append((all_cat_neg[i][(len(all_cat_neg[i])*65//100):(len(all_cat_neg[i])*80//100)]))
      test_cat_pos.append((all_cat_pos[i][(len(all_cat_pos[i])*80//100):]))
      test_cat_neg.append((all_cat_neg[i][(len(all_cat_neg[i])*80//100):]))
    train_cat_pos1 = np.array([ii for i in train_cat_pos for ii in i ])
    train_cat_neg1 = np.array([ii for i in train_cat_neg for ii in i ])
    valid_cat_pos1 = np.array([ii for i in valid_cat_pos for ii in i ])
    valid_cat_neg1 = np.array([ii for i in valid_cat_neg for ii in i ])
    test_cat_pos1 = np.array([ii for i in test_cat_pos for ii in i ])
    test_cat_neg1 = np.array([ii for i in test_cat_neg for ii in i ])
  elif itm == 6:
    # train_cat = np.array(np.vstack((train_cat_pos1,train_cat_neg1)))
    valid_cat = np.array(np.vstack((valid_cat_pos1,valid_cat_neg1)))
    test_cat = np.array(np.vstack((test_cat_pos1,test_cat_neg1)))
    # train_final = np.array(train_cat[:,:3])
    train_final = np.array(train_cat_pos1[:,:3])
  elif itm == 7:
    m1 = np.array(target).astype(np.float64)
    m2 = np.array(enzyme).astype(np.float64)
    m3 = np.array(pathway).astype(np.float64)
    m4 = np.array(smile).astype(np.float64)
    # print("m1 : ",len(m1)," m2 : ",len(m2)," m3 : ",len(m3)," m4 : ",len(m4))
    f_all_m1 = np.array(np.column_stack((drugs[:,0],m1)))
    f_all_m2 = np.array(np.column_stack((drugs[:,0],m2)))
    f_all_m3 = np.array(np.column_stack((drugs[:,0],m3)))
    f_all_m4 = np.array(np.column_stack((drugs[:,0],m4)))
    print(len(f_all_m1[:]),len(f_all_m1[0]))
  else:
    print("\n################# DDI copmleted ##################")
    print("################# featuers copmleted ##################")
    
print(test_cat_pos1.shape,valid_cat_pos1.shape,train_cat_pos1.shape,train_cat_pos1[0],valid_cat_pos1[0])
tr = []
print(" event >> valid : test >> whate events in valid : whate events in test ")
for i in range(572):
  s1 = len(train_cat_pos1[np.where(train_cat_pos1[:,1:]==i)])
  s2 = len(valid_cat_pos1[np.where(valid_cat_pos1[:,1:]==i)]) 
  s3 = len(test_cat_pos1[np.where(test_cat_pos1[:,1:]==i)])
  if s1 == 0:
    vid = valid_cat_pos1[np.where(valid_cat_pos1[:,1:]==i)[0],:].tolist()
    tst = test_cat_pos1[np.where(test_cat_pos1[:,1:]==i)[0],:].tolist()
    print(i," >>   ",s2," : ",s3 ," >> "
    ,np.unique(valid_cat_pos1[np.where(valid_cat_pos1[:,1:]==i)[0],0])
    ," : ",np.unique(test_cat_pos1[np.where(test_cat_pos1[:,1:]==i)[0],0])
    ," >>   ",vid," : ",tst )
    if not np.isnan(vid).all():
      for item in vid:
        tr.append(item)
    if not np.isnan(tst).all():
      for item in tst:
        tr.append(item)

print(" missing sample in train : " , tr)

print(" event >> test : valid")
for i in range(572):
  s4 = len(test_cat_neg1[np.where(test_cat_neg1[:,1:]==i)])
  s5 = len(valid_cat_neg1[np.where(valid_cat_neg1[:,1:]==i)])
  if s4 == 0 or s5 == 0:
    print(i," >>   ",s4," : ",s5 )

s1, s2, s3, s4, s5 ,s_all ,l_all ,persnt ,sal= [], [], [], [], [], [], [], [], 0
events = np.unique(full_pos[:,0])
for i in events:
  if i < 6:
    pp = (len(full_pos[np.where(full_pos[:,0]==i)])/len(full_pos))*100
    persnt.append(round(pp,1))
    l_all.append(""+str(round(pp,1))+"%")
    s_all.append(len(full_pos[np.where(full_pos[:,0]==i)]))
  else:
    sal += len(full_pos[np.where(full_pos[:,0]==i)])
  s1.append(len(train_cat_pos1[np.where(train_cat_pos1[:,0]==i)]))
  s2.append(len(valid_cat_pos1[np.where(valid_cat_pos1[:,0]==i)]))
  s3.append(len(test_cat_pos1[np.where(test_cat_pos1[:,0]==i)]))
  s4.append(len(test_cat_neg1[np.where(test_cat_neg1[:,0]==i)]))
  s5.append(len(valid_cat_neg1[np.where(valid_cat_neg1[:,0]==i)]))
  # print(i," >> ",s1," : ",s2 ," : ",s3 ," : ",s4 ," : ",s5 )
s_all.append(sal)
persnt.append(round(sal/len(full_pos)*100,1))
l_all.append(""+str(persnt[-1])+"%")

plt.title("number of samples in the events")
plt.plot(events, s1, label='data train_pos + ')
plt.plot(events, s2, label='data valid_pos + ')
plt.plot(events, s3, label='data test_pos + ')
plt.plot(events, s4, label='data test_neg - ')
plt.plot(events, s5, label='data valid_neg - ')
plt.xlabel('event')
plt.ylabel('number of samples')
plt.legend()
plt.show()
# print('\n',tr,'\n',vid,'\n',tst)
print(train_final.shape)
tr1 = np.array(tr)

# plt.title("number of samples in the events")
# plt.bar(l_all, s_all)
print(s_all," | sum : ",sum(s_all)," all : ",len(full_pos),"\n",persnt," | ",sum(persnt))

ddd = np.zeros((7)).tolist()
ddd[0] = 0.1
myexplode = ddd

labels = ["event "+str(i+1)+" : "+str(j) for i,j in enumerate(l_all)]
labels[-1] = "events 7-65 : "+str(l_all[-1])
# title = plt.title("number of samples in the events")
# title.set_ha("center")
plt.gca().axis("equal")
pie = plt.pie(persnt, labels = l_all, explode = myexplode, startangle=90)
plt.legend(pie[0],labels, bbox_to_anchor=(0.83,0.5), loc="center", fontsize=10, bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.6)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'event_pie.svg'
plt.savefig(image_name, format=image_format, dpi=1200)

print(tr1.shape, tr1)
train_final1 = np.concatenate((train_final,tr1[:,:3]))
print(train_final1.shape,train_final1[-1])
train_final = train_final1

df = pd.DataFrame(np.array(train_final))
df.to_csv('/DDI/data5/train.txt', header=None, index=None, sep=' ')
df = pd.DataFrame(np.array(valid_cat))
df.to_csv('/DDI/data5/valid.txt', header=None, index=None, sep=' ')
df = pd.DataFrame(np.array(test_cat))
df.to_csv('/DDI/data5/test.txt', header=None, index=None, sep=' ')

def write_f(a_f,path_f):
  print(a_f.shape)
  a,b = a_f.shape
  b = b-1
  with open(path_f, "w") as txt_file:
    csv.writer(txt_file, delimiter=' ').writerow([a,b])
    csv.writer(txt_file, delimiter=' ').writerows(a_f)

print(len(f_all_m1[:]),len(f_all_m1[0]), f_all_m1.shape)
f1 = write_f(f_all_m1,'/DDI/data5/featuers_m1.txt')
f2 = write_f(f_all_m2,'/DDI/data5/featuers_m2.txt')
f3 = write_f(f_all_m3,'/DDI/data5/featuers_m3.txt')
f4 = write_f(f_all_m4,'/DDI/data5/featuers_m4.txt')


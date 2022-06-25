from pandas import DataFrame
import numpy as np
import pandas as pd
import csv

x1=np.array(np.array(pd.read_csv("/DDI/data/final_modelssd1_d_32.csv", header=None , sep=',')).tolist())
x2=np.array(np.array(pd.read_csv("/DDI/data/final_modelssd2_d_32.csv", header=None , sep=',')).tolist())
x3=np.array(np.array(pd.read_csv("/DDI/data/final_modelssd3_d_32.csv", header=None , sep=',')).tolist())
x4=np.array(np.array(pd.read_csv("/DDI/data/final_modelssd4_d_32.csv", header=None , sep=',')).tolist())

def chang_to_array(s):
  # print(s.shape)
  final_model = []
  # print(final_model.shape)
  for i in s:
    ev = int(i[0])
    dr = int(i[1])
    h = i[2].replace('[',"")
    h = h.replace('...',"")
    h = h.replace('\n',"")
    h = h.replace(']',"")
    h = h.replace('  '," ")
    h = h.replace('  '," ")
    h = h.replace('  '," ")
    h = h.replace('  '," ")
    h = h.replace('  '," ")
    h = h.replace('  '," ")
    h = h.split(" ")
    # h.pop(0)
    for dd,d in enumerate(h):
      try:
        d = float(d)
      except:
        h.remove(d)
    h = np.array(h, dtype=np.float64)
    # print("ff : ",final_model[i,j])
    con = np.concatenate(([ev,dr],h))
    final_model.append(con)
  return np.array(final_model)

x1 = chang_to_array(x1)
print(x1.shape)
x2 = chang_to_array(x2)
print(x2.shape)
x3 = chang_to_array(x3)
print(x3.shape)
x4 = chang_to_array(x4)
print(x4.shape)

print(int(x1[0][1]))
print(len(x1[0,:]))
print(np.array(x1[0,:]))

def reduc_shape(m):
  r = []
  for i in range(572):
    try:
      s2=np.where(m[:,1]==i)[0]
      dd = m[s2[0],2:]
      for j in s2[1:]:
        # dd = np.mean((dd,m[j,2:]), axis=0)
        dd = np.concatenate((dd,m[j,2:]))
      r.append([i,dd])
    except:
      print("c")
  return np.array(r)

xx1 = np.array(reduc_shape(x1))
xx2 = np.array(reduc_shape(x2))
xx3 = np.array(reduc_shape(x3))
xx4 = np.array(reduc_shape(x4))

print("xx : ",xx1.shape)
print(len(xx1))
print(len(xx1[2][1]))
print(xx1[2])
print(xx1[336])
print(65*32)

def make_dic(x):
  s_dic = dict()
  for i in x:
    s_dic[i[0]]=i[1]
    # print(i[1][0])
  return s_dic

xs1 = (make_dic(xx1))
xs2 = (make_dic(xx2))
xs3 = (make_dic(xx3))
xs4 = (make_dic(xx4))
print(len(xs1[0]))

all_fectuer = []
all_fectuer.append(xs1)
all_fectuer.append(xs2)
all_fectuer.append(xs3)
all_fectuer.append(xs4)
all_fectuer = np.array(all_fectuer)
print(all_fectuer.shape)
print(len(all_fectuer[0]))
print(len(all_fectuer[0][0]))
print(all_fectuer[0][0][0])
# print(all_fectuer[0][336][0])
print(type(all_fectuer[0]))

full_pos=np.array(np.array(pd.read_csv("/DDI/full_pos.txt", header=None , sep=' ')).tolist())
print(full_pos.shape)

DDI = full_pos[:,1:3]
print(DDI.shape)
print(65*570)

print(all_fectuer.shape)
f_i1 = all_fectuer[0]
f_i2 = all_fectuer[1]
f_i3 = all_fectuer[2]
f_i4 = all_fectuer[3]

new_feature1 = ( np.array(np.multiply(f_i1[d[0]],f_i1[d[1]])).tolist() for d in (DDI) )
new_feature2 = ( np.array(np.multiply(f_i2[d[0]],f_i2[d[1]])).tolist() for d in (DDI) )
new_feature3 = ( np.array(np.multiply(f_i3[d[0]],f_i3[d[1]])).tolist() for d in (DDI) )
new_feature4 = ( np.array(np.multiply(f_i4[d[0]],f_i4[d[1]])).tolist() for d in (DDI) )

df = pd.DataFrame(np.array(list(new_feature1)))
df.to_csv('/DDI/data5/t_c_m_1_32.txt', header=None, index=None, sep=' ')
df = pd.DataFrame(np.array(list(new_feature2)))
df.to_csv('/DDI/data5/t_c_m_2_32.txt', header=None, index=None, sep=' ')
df = pd.DataFrame(np.array(list(new_feature3)))
df.to_csv('/DDI/data5/t_c_m_3_32.txt', header=None, index=None, sep=' ')
df = pd.DataFrame(np.array(list(new_feature4)))
df.to_csv('/DDI/data5/t_c_m_4_32.txt', header=None, index=None, sep=' ')

print(len(all_fectuer[0][0]),len(DDI))
full_pos = 0
x1 ,x2 ,x3 ,x4 ,xx1 ,xx2 ,xx3 ,xx4,xs1 ,xs2 ,xs3 ,xs4 = 0,0,0,0,0,0,0,0,0,0,0,0
full_dataframe,f_dataframe,featuers,drugs,a,x=0,0,0,0,0,0


import pandas as pd
import os
import numpy as np
import scipy.io
df=pd.read_csv('beam.CSV')

target_cols=df.columns.values


list=[val for val in df['Length']]
list=np.array(list)

#print("x: ",np.shape(list))
list_sol=[
     df[x]  for x in target_cols if "Wn" in x
        ]
n=len(df)
m=len(target_cols)
tmpList=[]

i=0
print(m,n)
while i < n:
    tmp=[]
    for x in target_cols :
        
        if  "Wn" in x:
            tmp.append(df[x][i])
            # print(i,df[x][i])
    tmpList.append(tmp)
    i=i+1
usol=np.array(tmpList)


print("usol.shape",usol.shape)
usol_range=usol.shape[1]
t=[]

for i in range(usol_range):
    tmp=i/usol_range
    t.append(tmp)

t=np.array(t)


print(usol.shape)

y=[val for val in df['Thickness']]
y=np.array(y)
     
dico={"x":list,
      "usol":usol,
      "y":y}
scipy.io.savemat('beam_cantilever_100.mat',dico,oned_as='column')






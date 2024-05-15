import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import data as tf_data
import keras
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  # Normalized data to range from (0,1)
from sklearn.metrics import (
    precision_recall_curve,
    plot_precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)


def tide_finder(df, td, year): # finds the direction and percentage of the tide change for each buoy reading
    Dirs = [] # Direction
    t_str = [] # Percentage which should correspond to the strenght of the tide change
    m = 1
    ind_save = 0 # necessary for counting indices over the year
    print("Starting tide estimator for", year)
    while m < 13:
        print("Working on month", m)
        d = 1
        boo1 = df['Month']==m # Segments data for each month
        t0 = df[boo1]
        ld = max(t0['Day']) # Last day of that month
        while d <= ld: # Generic method for segmenting data for the day
            boo2 = t0['Day']==d
            t2 = t0[boo2]
            Zmins = [t2['Hour']*60+t2['Min']][0] # This is for buoy data calcs
            boo1 = td['Month']==m # this and below is for tide data 
            boo2 = td['Day']==d 
            t1 = td[boo1] 
            t2 = t1[boo2] # dataframe for the designated day
            #start_tide = t2['S'][:1].to_list()[0] 
            #end_tide = t2['S'][-1:].to_list()[0]
            # Finds tide turning points for the last tide the day before and the first tide the next day
            # First finds the next days tide point
            if d < ld: # Concerns majority of cases
                boo3 = td['Day']==d+1
                t3 = t1[boo3][:1]
            elif d == ld and m!=12: # Accounts for last day of the month edge case
                boo3 = td['Month']==m+1
                boo4 = td['Day']==1
                t25 = td[boo3]
                t3 = t25[boo4][:1]
            else: # Accounts for Dec 31 edge case
                boo3 = td['Month']==1
                boo4 = td['Day']==1
                #boo5 = td['S']!=end_tide
                t233 = td[boo3]
                t266 = t233[boo4][:1]
                t3 = t266 #[boo5]
            
            # Second finds yesterdays final tide point    
            if d > 1: # Concerns majority of cases
                boo3 = td['Day']==d-1
                t4 = t1[boo3][-1:]
            elif d==1 and m!=1: # Accounts for first day of the month edge case
                boo3 = td['Month']==m-1
                t25 = td[boo3]
                tld = max(t25['Day'])
                boo4 = t25['Day']==tld
                t4 = t25[boo4]
            else: # Accounts for Jan. 1 edge case
                boo3 = td['Month']==12
                boo4 = td['Day']==31
                #boo5 = td['S']!=start_tide
                t233 = td[boo3]
                t266 = t233[boo4][-1:]
                t4 = t266 #[boo5]
                
            t3['TM'] = round((t3['Hour']+24)*60+t3['Min'],2) # Day after tide time
            t4['TM'] = -round((24-t4['Hour'])*60-t4['Min'],2) # Day before tide time
            t2['TM'] = round(t2['Hour']*60+t2['Min'],2)
            t2 = pd.concat([t4,t2,t3])
            Tmins = t2['TM']
            for i in range(0,len(Zmins)):
                diffs = round(Tmins - Zmins[i+ind_save],2)
                point1 = min(diffs[diffs>0])
                point2 = min(abs(diffs[diffs<0]))
                targ = round(point1 + Zmins[i+ind_save],2)
                prev = round(Zmins[i+ind_save] - point2,2)
                new_t = t2[t2['TM']==targ]
                prev_t = t2[t2['TM']==prev]
                ntp = new_t['S'].to_list()[0]
                ptp = prev_t['S'].to_list()[0]
                nth = new_t['H'].to_list()[0]
                pth = prev_t['H'].to_list()[0]
                lims = abs(nth - pth)
                if ptp=='L':
                    Dirs.append(0)
                if ptp=='H':
                    Dirs.append(1)
            ind_save += len(Zmins)
            d += 1
            
        m += 1
        
    rpt = 0
    for x in range(1,len(Dirs)):
         # reading per tide direction
        templist = []
        if Dirs[x] == Dirs[x-1]:
            rpt += 1
        else:
            rpt += 1
            ttp = rpt/2 # time to peak
            y = 1
            if ttp == int(ttp):
                while y <= ttp:
                    templist.append(y/ttp)
                    y+=1
                while y > 0:
                    y-=1
                    templist.append(y/ttp)
            if ttp != int(ttp):
                ttp += 0.5
                while y <= ttp:
                    templist.append(y/ttp)
                    y+=1
                y-=1
                while y > 0:
                    y-=1
                    templist.append(y/ttp)
            rpt = 0


        if x == len(Dirs)-1:
            rpt += 1
            ttp = rpt/2 # time to peak
            y = 1
            if ttp == int(ttp):
                while y <= ttp:
                    templist.append(y/ttp)
                    y+=1
                while y > 0:
                    y-=1
                    templist.append(y/ttp)
            if ttp != int(ttp):
                ttp += 0.5
                while y <= ttp:
                    templist.append(y/ttp)
                    y+=1
                y-=1
                while y > 0:
                    y-=1
                    templist.append(y/ttp)

        for val in templist[:-1]:
            t_str.append(val)


                    
            
        
    lib = {'Dirs':Dirs,'STR':t_str}
    newdat = pd.DataFrame(lib)
    print("Successfully completed tide estimation for ", year)
    return lib

def data_org(buoy, tnames):
    df = pd.read_csv(buoy)
    init = df.keys()[0]
    use_data = df[init][1:]
    
    if slim:
        for val in use_data:
            sep_vals = val.split(' ')
            sift = filter(lambda item: item != '', sep_vals)
            mdata = list(sift)
            tnames['Month'].append(int(mdata[1]))
            tnames['Day'].append(int(mdata[2]))
            tnames['Hour'].append(int(mdata[3]))
            tnames['Min'].append(int(mdata[4]))
            tnames[anames[10]].append(float(mdata[8]))
            tnames[anames[13]].append(float(mdata[9]))
            tnames[anames[16]].append(float(mdata[10]))
            tnames[anames[17]].append(int(mdata[11]))
            
    else:
        for val in use_data:
            sep_vals = val.split(' ')
            sift = filter(lambda item: item != '', sep_vals)
            mdata = list(sift)
            tnames['Month'].append(int(mdata[1]))
            tnames['Day'].append(int(mdata[2]))
            tnames['Hour'].append(int(mdata[3]))
            tnames['Min'].append(int(mdata[4]))
            tnames['WDIR'].append(float(mdata[5]))
            tnames['WSPD'].append(float(mdata[6]))
            tnames['GST'].append(float(mdata[7]))
            tnames['WVHT'].append(float(mdata[8]))
            tnames['DPD'].append(float(mdata[9]))
            tnames['APD'].append(float(mdata[10]))
            tnames['MWD'].append(float(mdata[11]))
            tnames['PRES'].append(float(mdata[12]))


    
    A = pd.DataFrame(tnames)
    return A


tidesource = ['Tides/2023RJsannual.txt','Tides/2022RJsannual.txt','Tides/2023RJsannual.txt','Tides/2022RJsannual.txt'
              ,'Tides/2023SCannual.txt','Tides/2022SMannual.txt','Tides/2023SMannual.txt',
             'Tides/2022LJannual.txt','Tides/2023LJannual.txt']
tide_data = []
for item in tidesource:
    temp1 = pd.read_csv(item)
    temp2 = temp1[temp1.keys()[0]]
    for val in temp2:
        tide_data.append(val)


tide_inf = {'Month':[],'Day':[],'Hour':[],'Min':[],'H':[],'S':[],'TH':[]}
for date in tide_data:
    sep_vals = date.split('/')
    tide_inf['Month'].append(int(sep_vals[1]))
    sep_vals = sep_vals[2].split('\t')
    tide_inf['Day'].append(int(sep_vals[0]))
    hm = sep_vals[2].split(':')
    tide_inf['Hour'].append(int(hm[0]))
    tide_inf['Min'].append(int(hm[1]))
    tide_inf['H'].append(float(sep_vals[3]))
    tide_inf['S'].append(sep_vals[-1])
    tide_inf['TH'].append(0)

tidedf = pd.DataFrame(tide_inf)

refdat = 'SPS23.txt'
df = pd.read_csv(refdat)

anames = df.keys()[0].split(' ')
cnames = {'Month':[],'Day':[],'Hour':[],'Min':[],
          anames[10]:[],anames[13]:[],anames[16]:[],anames[17]:[]}

slim=True
buoy1 = 'SPS23.txt'
buoy2 = 'SPS22.txt'
buoy3 = 'SP23.txt'
buoy4 = 'SP22.txt'
buoy5 = 'RBN23.txt'
buoy6 = 'SM22.txt'
buoy7 = 'SM23.txt'
buoy8 = 'LJ22.txt'
buoy9 = 'LJ23.txt'
bl = [buoy1,buoy2,buoy3,buoy4,buoy5,buoy6,buoy7,buoy8,buoy9]

fdata = {'Month':[],'Day':[],'Hour':[],'Min':[],
      anames[10]:[],anames[13]:[],anames[16]:[],anames[17]:[]}

fdata = pd.DataFrame(fdata)


tinf = {'Dirs':[],'STR':[]}
tinf = pd.DataFrame(tinf)


for buoy in bl:
    cnames = {'Month':[],'Day':[],'Hour':[],'Min':[],
          anames[10]:[],anames[13]:[],anames[16]:[],anames[17]:[]}
    shell = pd.DataFrame(cnames)
    Z = data_org(buoy,cnames)
    tdat = tide_finder(Z,tidedf,buoy)
    body = pd.concat([shell,Z])
    ttide = pd.DataFrame(tdat)
    tinf = pd.concat([tinf,ttide],ignore_index=True)
    fdata = pd.concat([fdata,body],ignore_index=True)
  
fdata['Dirs'] = tinf['Dirs']
fdata['STR'] = tinf['STR']
key_data = fdata['WVHT']
fdata = fdata.drop(['WVHT'],axis=1)

-----------------------------------------------------------

slim=False
source1 = 'SSR23.txt'
source2 = 'SSR22.txt'
source3 = 'SSR21.txt'
source4 = 'SSR20.txt'

source = [source1,source2,source3,source4]


cnames = {'Month':[],'Day':[],'Hour':[],'Min':[],'WDIR':[],'WSPD':[],'GST':[],'WVHT':[],'DPD':[],'APD':[],
         'MWD':[],'PRES':[]}


fdata = pd.DataFrame(cnames)
for year in source:
    shell=pd.DataFrame(cnames)
    temp = data_org(year,cnames)
    fdata = pd.concat([fdata,temp],ignore_index=True)


timedat = fdata[['Month','Day','Hour','Min']]
fdata = fdata[fdata['WVHT']!=99]
y = fdata['WVHT']
fdata = fdata.drop(['WVHT','Month','Day','Hour','Min'],axis=1)


fd = fdata

y = key_data
y = round(y,0)
y = pd.DataFrame(y)
mode=False
cats = ['WVHT1','WVHT2','WVHT3']

# i = 0
# for val in y['WVHT']:
#     if val==99:
#         y.drop(i)
#         fd.drop(i)
#     i += 1


    
fd = fd.reset_index(drop=True)
y = y.reset_index(drop=True)

print(len(fd),len(y))
    
c = []
if mode:
    for cat in cats:
        j = 0
        for val in fd[cat]:
            if val==99:
                c.append(j)
            j += 1
            
# else:
#     j = 0
#     for val in fd['WVHT']:
#         if val==99:
#             c.append(j)
#         j += 1
        
        
# for val in c:
#     fd = fd.drop(val)
#     y = y.drop(val)
    
fd = fd.reset_index(drop=True)
y = y.reset_index(drop=True)
        
#FEATURE_NAMES = fdata.keys()

seed = 8


X_train, X_test, Y_train, Y_test = train_test_split(fd, y, test_size=0.15,random_state=seed)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


shap = X_train.shape
numBranches = shap[1]
shap

shap2 = Y_test.shape
shap2

from xgboost import XGBClassifier 
mod = XGBClassifier()
mod.fit(X_train,Y_train)
  
# accuracy on X_test 
accuracy = mod.score(X_test, Y_test) 
print(accuracy)
  
# creating a confusion matrix 
y_pred = mod.predict(X_test)  
confusion_matrix(Y_test, y_pred) 

print(classification_report(Y_test,y_pred))

    




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def build_simple_model():
    op = keras.optimizers.Adam()
    model = Sequential
    model.add(Dense(5,input_shape=5,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=op,metrics=[])
    return model

def tide_finder(df, td, year): 
    Dirs = []
    Tide = []
    m = 1
    ind_save = 0
    print("Starting tide estimator for", year)
    while m < 13:
        print("Working on month", m)
        d = 1
        boo1 = df['Month']==m
        t0 = df[boo1]
        ld = max(t0['Day'])
        while d <= ld:
            boo2 = t0['Day']==d
            t2 = t0[boo2]
            Zmins = [t2['Hour']+t2['Min']/60][0]
            boo1 = td['Month']==m
            boo2 = td['Day']==d
            t1 = td[boo1]
            t2 = t1[boo2]
            start_tide = t2['S'][:1].to_list()[0] 
            end_tide = t2['S'][-1:].to_list()[0]
            if d < ld:
                boo3 = td['Day']==d+1
                t3 = t1[boo3][:1]
            elif d == ld and m!=12:
                boo3 = td['Month']==m+1
                boo4 = td['Day']==1
                t25 = td[boo3]
                t3 = t25[boo4][:1]
            else: 
                boo3 = td['Month']==1
                boo4 = td['Day']==1
                boo5 = td['S']!=end_tide
                t233 = td[boo3]
                t266 = t233[boo4][:2]
                t3 = t266[boo5]
                
            if d > 1:
                boo3 = td['Day']==d-1
                t4 = t1[boo3][-1:]
            elif d==1 and m!=1:
                boo3 = td['Month']==m-1
                t25 = td[boo3]
                tld = max(t25['Day'])
                boo4 = t25['Day']==tld
                t4 = t25[boo4]
            else:
                boo3 = td['Month']==12
                boo4 = td['Day']==31
                boo5 = td['S']!=start_tide
                t233 = td[boo3]
                t266 = t233[boo4][-2:]
                t4 = t266[boo5]
                
    
            t3['TH'] = round(t3['Hour']+24,2)
            t4['TH'] = round(t4['Hour']-24,2)
            t2['TH'] = round(t2['Hour']+t2['Min']/60,2)
            t2 = pd.concat([t4,t2,t3])
            Tmins = t2['TH']
            for i in range(0,len(Zmins)):
                diffs = round(Tmins - Zmins[i+ind_save],2)
                point1 = min(diffs[diffs>0])
                point2 = min(abs(diffs[diffs<0]))
                targ = round(point1 + Zmins[i+ind_save],2)
                prev = round(Zmins[i+ind_save] - point2,2)
                new_t = t2[t2['TH']==targ]
                prev_t = t2[t2['TH']==prev]
                ntp = new_t['S'].to_list()[0]
                ptp = prev_t['S'].to_list()[0]
                nth = new_t['H'].to_list()[0]
                pth = prev_t['H'].to_list()[0]
                lims = abs(nth - pth)
                if ptp=='L':
                    Dirs.append(0)
                    etide = pth + point2*lims/12
                if ptp=='H':
                    Dirs.append(1)
                    etide = pth - point2*lims/12
            
                Tide.append(round(etide,2))
            ind_save += len(Zmins)
            d += 1
            
        m += 1
    lib = {'Tide':Tide,'Dirs':Dirs}
    newdat = pd.DataFrame(lib)
    print("Successfully completed tide estimation for ", year)
    return newdat


seed = 8
df = pd.read_csv('46253his.txt')
anames = df.keys()[0].split(' ')
shell = df.keys()[0]
use_data = df[shell][1:]

tide_file = pd.read_csv('Tides/2023RJsannual.txt')
tide_data = tide_file[tide_file.keys()[0]]

cnames = {'Month':[],'Day':[],'Hour':[],'Min':[],
          anames[10]:[],anames[13]:[],anames[16]:[],anames[17]:[]}
tide_inf = {'Month':[],'Day':[],'Hour':[],'Min':[],'H':[],'S':[],'TH':[]}
for val in use_data:
    sep_vals = val.split(' ')
    sift = filter(lambda item: item != '', sep_vals)
    mdata = list(sift)
    cnames['Month'].append(int(mdata[1]))
    cnames['Day'].append(int(mdata[2]))
    cnames['Hour'].append(int(mdata[3]))
    cnames['Min'].append(int(mdata[4]))
    cnames[anames[10]].append(float(mdata[8]))
    cnames[anames[13]].append(float(mdata[9]))
    cnames[anames[16]].append(float(mdata[10]))
    cnames[anames[17]].append(int(mdata[11]))

    
Z = pd.DataFrame(cnames)

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


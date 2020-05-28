#!/usr/bin/env python
#convert stream to signature
def read_name(dir_name):
  a    = []
  num_ex = 3 #number of levels should be greater than num_ex
  j = 0
  j0 = 0
  for file in glob.glob(dir_name):
    nc = netCDF4.Dataset(file, 'r')
    n_levels = nc.dimensions['N_LEVELS'].size
    n_param  = nc.dimensions['N_PARAM'].size
    pres = nc.variables['PRES'][:]
    pres_max = np.amax(pres)
    pres_min = np.amin(pres)
    dpres = pres_max - pres_min
    if dpres > 1000. and n_levels>=num_ex and n_param >= 3:
      j += 1
      a.append(os.path.basename(file))
    else:
      j0 += 1
  nsample = j
  print(nsample, "samples", j0, "excluded")
  return a
def draw_bar(file,j0,j1,x):
  #plot figure
  fig = plt.figure()
  plt.title('Amplitude of Iterated Integrals')
  plt.xlabel('Index Number of Iterated Integral')
  plt.ylabel('Amplitude or Weight')
  plt.yscale('log')
  plt.xlim(xmax = 40, xmin = 0)
  plt.ylim(ymax = 2e1, ymin = 1e-3)
#  plt.hold(True)
  plt.scatter(np.arange(40)+0.3,np.abs(w[0:40]),color='g',label='|weight|')
  plt.bar(np.arange(40)    ,np.abs(x[j0][0:40]),alpha=0.4,color='b',width=0.3,label='sample with y=0')
  plt.bar(np.arange(40)+0.3,np.abs(x[j1][0:40]),alpha=0.4,color='r',width=0.3,label='sample with y=1')
  plt.legend() 
  plt.savefig(file)
  plt.close()

  
def draw_histo_log(file,m1,y0,y1):
  #plot figure
  fig = plt.figure()
  plt.title('Histogram for Prediction with Profile Shapes w digit '+m1)
  plt.xlabel('Predicted Y-value')
  plt.ylabel('Frequency')
  plt.ylim([8e-1,1e3])
  plt.hist(y0,bins=40,range=(-0.5,1.5),alpha=0.5, histtype='stepfilled', color='b',log=True)
  plt.hist(y1,bins=40,range=(-0.5,1.5),alpha=0.5, histtype='stepfilled', color='r',log=True)
  plt.savefig(file)
  plt.close()

def draw_histo(file,m1,y0,y1):
  #plot figure
  fig = plt.figure()
  plt.title('Histogram for Prediction with Profile Shapes w digit '+m1)
  plt.xlabel('Predicted Y-value')
  plt.ylabel('Frequency')
  plt.ylim([0,900])
  plt.hist(y0,bins=40,range=(-0.5,1.5),alpha=0.5, histtype='stepfilled', color='b')
  plt.hist(y1,bins=40,range=(-0.5,1.5),alpha=0.5, histtype='stepfilled', color='r')
  plt.savefig(file)
  plt.close()
  
def draw_histos(file,m1,y0,y1):
  #plot figure
  fig = plt.figure()
  plt.title('Histogram for Prediction with Profile Shapes w digit '+m1)
  plt.xlabel('Predicted Y-value')
  plt.ylabel('Frequency')
  plt.ylim([0,900])
  plt.hist([y0,y1],bins=40,range=(-0.5,1.5), alpha=0.5, stacked=True, color=['b','r'])
  plt.savefig(file)
  plt.close()

def write_pred(file0,file1,a,y,yp):
  #print "Prediction:"
  f0 = open(file0, 'w') 
  f1 = open(file1, 'w')
  y1 = []
  y0 = []
  nsample = len(y)
  for j in range(nsample):
    if y[j][0] > 0:
      f1.write(str(yp[j])+' '+str(y[j][0])+' '+a[j]+' \n')
      y1.append(yp[j])
    else:
      f0.write(str(yp[j])+' '+str(y[j][0])+' '+a[j]+' \n')
      y0.append(yp[j])
  f0.close() 
  f1.close()
  return y0,y1
def write_score(y,yp):
  n_ALL = len(y)
  n_NG = 0
  for i in range(n_ALL):
    if (y[i][0] == 1 and yp[i] < 0.5):
      n_NG += 1
    if (y[i][0] == 0 and yp[i] > 0.5):
      n_NG += 1
  n_OK = 0
  for i in range(n_ALL):
    if (y[i][0] == 1 and yp[i] > 0.5):
      n_OK += 1
    if (y[i][0] == 0 and yp[i] < 0.5):
      n_OK += 1
  print("Error rate:", float(n_NG)/float(n_ALL), "=", n_NG, "/", n_ALL)
  print("Valid rate:", float(n_OK)/float(n_ALL), "=", n_OK, "/", n_ALL)

from sklearn import linear_model
import numpy as np
import csv, glob, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#argvs = sys.argv
dim = 3      #dimension of strm
ord = 6      #truncation of signature
#scale = float(1)/float(argvs[1])
scale = 1.0
a=1.0e-5
reg = linear_model.Lasso(alpha = a,tol=0.01,max_iter=5000,copy_X=True,fit_intercept=True,selection='random')
scaler = StandardScaler()

num  = ((2*dim)**(ord+1)-1)/(2*dim-1)

nn = int(10)
nn2 = int(1)
ini = True
for m in range(nn):
   m1 = str(m)
   y_name = "yi_" + m1 + ".npy"
   if ini == True:
     y = np.load(y_name)
     print("data loaded", m, y.shape[0], y.shape[1])
     ini = False
   else:
     yt = np.load(y_name)
     y  = np.concatenate([y,yt],axis=0)
     print("data loaded", m, y.shape[0], y.shape[1])
x = np.zeros([int(y.shape[0]),int(num)])
ini = True
a = []
for m in range(nn):
   m1 = str(m)
   x_name = "x_" + m1 + ".npy"
   y_name = "yi_" + m1 + ".npy"
#   dir_name = "ArgoData/*/profiles/D*_??" + m1 + ".nc" 
#   dir_name = "ArgoData/490168?/profiles/D*_??" + m1 + ".nc" 
   dir_name = "ArgoData/290028?/profiles/D*_??" + m1 + ".nc" 
   a.extend(read_name(dir_name))
   if ini == True:
     n0 = 0
     y = np.load(y_name)
     x[n0:y.shape[0],:] = np.load(x_name)
     print("data loaded", m, n0,"-", y.shape[0], x.shape[1])
     ini = False
   else:
     n0 = y.shape[0]
     yt = np.load(y_name)
     y  = np.concatenate([y,yt],axis=0)
     x[n0:y.shape[0],:] = np.load(x_name)
     print("data loaded", m, n0,"-", y.shape[0], x.shape[1])
for mm in range(nn2):
 print("Experiment for predicting ", mm)
 m1 = str(mm)
 idx = np.array(range(x.shape[0]))
 (x_train, x_test,y_train, y_test, idx_train, idx_test) \
   = train_test_split(x, y, idx, test_size=0.6,random_state=mm)
 print("std before",x_train)
 scaler.fit(x_train)
 x_train = scaler.transform(x_train)
 print("std after",x_train)
 reg.fit(x_train,y_train)
 ws = reg.sparse_coef_
 w = reg.coef_
 print("Coefficients:")
 print(ws[:][:])
 n0_ws = 0
 n1_ws = 0
 for i in range(ws.shape[1]):
   if (ws[0,i] == 0):
     n0_ws += 1
   else:
     n1_ws += 1
 print("non-zero ws ",n1_ws, "zero ws ", n0_ws)
 file = 'Sig' + str(ord) + "_" + m1 + ".eps"
 draw_bar(file,0,1,x_train)
 print(a, reg.score(x_train,y_train))

 yp = reg.predict(x_train)
 file_c0 = 'l0_' + m1 + '.txt'
 file_c1 = 'l1_' + m1 + '.txt'
 a_train = []
 for i in idx_train:
   a_train.append(a[i])
 y0,y1 = write_pred(file_c0,file_c1,a_train,y_train,yp)
 print("Score (train):")
 print(a, reg.score(x_train,y_train))
 write_score(y_train,yp)

 x_test = scaler.transform(x_test)
 yp = reg.predict(x_test)
 file_c0 = 'c0_' + m1 + '.txt'
 file_c1 = 'c1_' + m1 + '.txt'
 a_test = []
 for i in idx_test:
   a_test.append(a[i])
 y0,y1 = write_pred(file_c0,file_c1,a_test,y_test,yp)
 file = 'Histo' + str(ord) + "_" + m1 + ".eps"
 draw_histo(file,m1,y0,y1)
 file = 'HistoLog' + str(ord) + "_" + m1 + ".eps"
 draw_histo_log(file,m1,y0,y1)
 file = 'Histos' + str(ord) + "_" + m1 + ".eps"
 draw_histos(file,m1,y0,y1)
 print("Score:")
 print(a, reg.score(x_test,y_test))
 write_score(y_test,yp)


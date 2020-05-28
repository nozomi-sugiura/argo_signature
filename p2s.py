#!/usr/bin/env python
#convert stream to signature
def ll_trans(ll,pres,psal,temp):
  d = 3
  l     = np.array([0   for i in range(2*d)])
  ld    = np.array([0   for i in range(2*d)])
  z     = np.array([0.0 for i in range(2*d)])
  z_old = np.array([0.0 for i in range(2*d)])
  strm = []
  tt = 0
  for t0 in range(30000):
    z[0] = pres[0][t0]
    z[1] = psal[0][t0]
    z[2] = temp[0][t0]
    z[3] = pres[0][t0]
    z[4] = psal[0][t0]
    z[5] = temp[0][t0]
    if not (np.isnan(z[0]) or np.isnan(z[1]) or np.isnan(z[2])):
      strm.append([z[0]*scale/float(2000),(z[1]-35.)*scale/float(2),z[2]*scale/float(20),\
                   z[3]*scale/float(2000),(z[4]-35.)*scale/float(2),z[5]*scale/float(20)])
#      print("{0:-11d} {1:-11.4f} {2:-11.4f} {3:-11.4f} {4:-11.4f} {5:-11.4f} {6:-11.4f}"\
#            .format(t0, z[0],z[1],z[2],z[3],z[4],z[5]))
      tt += 1
      break
  for i in range(2*d):
    z_old[i]    = z[i]
  for t in range(t0+1,30000):
    for i in range(2*d):
      l[i] = (t-i+2*d-1)//(2*d)
    if (l[2*d-1] < ll):
      for i in range(2*d):
        ld[i] = min(l[i],ll-1)
      z[0] = pres[0][ld[0]]
      z[1] = psal[0][ld[1]]
      z[2] = temp[0][ld[2]]
      z[3] = pres[0][ld[3]]
      z[4] = psal[0][ld[4]]
      z[5] = temp[0][ld[5]]
      for i in range(2*d):
        if (np.isnan(z[i])):
          z[i]  = z_old[i]
      strm.append([z[0]*scale/float(2000),(z[1]-35.)*scale/float(2),z[2]*scale/float(20),\
                   z[3]*scale/float(2000),(z[4]-35.)*scale/float(2),z[5]*scale/float(20)])
#      print("{0:-11d} {1:-11.4f} {2:-11.4f} {3:-11.4f} {4:-11.4f} {5:-11.4f} {6:-11.4f}"\
#            .format(t, z[0],z[1],z[2],z[3],z[4],z[5]))
      for i in range(2*d):
        z_old[i]  = z[i]
      tt += 1
    else:
      break
  return strm,tt

def read_strm(dir_name, scale):
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
    else:
      j0 += 1
  nsample = j
  print( nsample, "samples", j0, "excluded")
  num  = ((2*dim)**(ord+1)-1)/(2*dim-1)
  x    = np.array([[0.0 for i in range(int(num))] for j in range(int(nsample))])
  y    = np.array([[0.0 for i in range(int(1))]   for j in range(int(nsample))])
  print( "initialized x:", type(x), x.shape[0], x.shape[1])
  print( "initialized y:", type(y), y.shape[0], y.shape[1])
  j = 0
  for file in glob.glob(dir_name):
    nc = netCDF4.Dataset(file, 'r')
    n_levels = nc.dimensions['N_LEVELS'].size
    n_param  = nc.dimensions['N_PARAM'].size
    pres = nc.variables['PRES'][:]
    pres_max = np.amax(pres)
    pres_min = np.amin(pres)
    dpres = pres_max - pres_min
    if dpres > 1000. and n_levels>=num_ex and n_param >= 3:
      pres = nc.variables['PRES'][:]
      psal = nc.variables['PSAL'][:]
      temp = nc.variables['TEMP'][:]
      pres = np.ma.filled(pres.astype(float), np.nan)
      psal = np.ma.filled(psal.astype(float), np.nan)
      temp = np.ma.filled(temp.astype(float), np.nan)
      pres_qc = nc.variables['PRES_ADJUSTED_QC'][:]
      psal_qc = nc.variables['PSAL_ADJUSTED_QC'][:]
      temp_qc = nc.variables['TEMP_ADJUSTED_QC'][:]
      ll = int(pres.shape[1])
      qc = 1
      strm,tt = ll_trans(ll,pres,psal,temp)
      print( "included", n_levels, n_param, os.path.basename(file), j, dpres, tt)
#      print( j, strm)
      for l in range(ll):
        if (np.int(pres_qc[0][l])-4)*(np.int(psal_qc[0][l])-4)*(np.int(temp_qc[0][l])-4) == 0:
          qc = 0
      x[j][:] = ts.stream2sig(np.array(strm), ord)
      if qc == 1:
        y[j][0] = float(1)
      else:
        y[j][0] = float(0)
      j += 1
    else:
      print( "excluded", n_levels, n_param, os.path.basename(file), j, dpres)
  return x, y

import esig.tosig as ts
import numpy as np
import csv, glob, sys, os
import netCDF4
import boost
#argvs = sys.argv
dim = 3      #dimension of strm
ord = 6      #truncation of signature
#scale = float(1)/float(argvs[1])
scale = 1.0
#ms = int(argvs[2])
for m in range(10):
  m1 = str(m)
#  dir_name = "ArgoData/*/profiles/D*_??" + m1 + ".nc" 
#  dir_name = "ArgoData/490168?/profiles/D*_??" + m1 + ".nc" 
  dir_name = "ArgoData/290028?/profiles/D*_??" + m1 + ".nc" 
  x_name = "x_" + m1
  y_name = "yi_" + m1
  x,y = read_strm(dir_name, scale)
  print( "start saving data")
  np.save(x_name,x)
  np.save(y_name,y)
  





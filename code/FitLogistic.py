# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:38:40 2020

@author: jru
"""

from numba import jit
import numpy as npy
import NLLSfit

@jit(nopython=True)
def getAmpOffset(function,data):
    sumx2=npy.sum(function*function)
    sumx=npy.sum(function)
    sumy=npy.sum(data)
    sumxy=npy.sum(function*data)
    dlength=float(len(data))
    if(sumx2>0.0):
        divider=dlength*sumx2-sumx*sumx
        off=(sumx2*sumy-sumx*sumxy)/divider
        amp=(dlength*sumxy-sumx*sumy)/divider
    else:
        amp=0.0
        off=sumy/dlength
    return amp,off

@jit(nopython=True)
def getLogistic(xvals,params):
    return params[0]+params[1]/(1.0+npy.exp(-params[3]*(xvals-params[2])))

@jit(nopython=True)
def getAmpOffsetC2(func,data,amp,off):
    resid=amp*func+off-data
    c2=npy.sum(resid*resid)
    return c2/float(len(data)-2)

@jit(nopython=True)
def initKd(xvals,data,minkd1,maxkd1,guessn):
    minc2=-1.0
    minkd=minkd1
    minamp=0.0
    minoff=0.0
    kd=minkd
    while(kd<=maxkd1):
        fitfunc=getLogistic(xvals,npy.array([0.0,1.0,kd,guessn]))
        amp,off=getAmpOffset(fitfunc,data)
        c2=getAmpOffsetC2(fitfunc,data,amp,off)
        if(minc2<0.0 or c2<minc2):
            minc2=c2
            minkd=kd
            minamp=amp
            minoff=off
        kd*=1.05
    return [minkd,minc2,minamp,minoff]

def runFit(xvals,data,minkd,maxkd,guessn,fampmin):
    def fitfunc(params):
        return getLogistic(xvals,params)
       
    guessparams=initKd(xvals,data,minkd,maxkd,guessn)
    fitclass=NLLSfit.NLLSFit(fitfunc)
    if(guessparams[1]<0.0):
        guessparams[1]=max(data)-min(data)
    params=[0.0,guessparams[2],guessparams[0],guessn]
    constraints=[[-0.1*params[1],fampmin*params[1],minkd,0.1],[0.1*params[1],5.0*params[1],maxkd,2.0]]
    fixes=[0,0,0,0]
    iterations,c2,params,fit=fitclass.fitData(params,fixes,constraints,data,verbose=False)
    return iterations,c2,params,fit
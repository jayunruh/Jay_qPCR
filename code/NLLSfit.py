# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:35:11 2020

@author: jru
"""
import numpy as npy

class NLLSFit():

    def __init__(self,fitfunc,toler=0.0001,maxiter=50,lambdaval=0.1):
        self.toler=npy.double(toler)
        self.maxiter=maxiter
        self.lambdaval=npy.double(lambdaval)
        self.fitfunc=fitfunc
        self.dx=npy.double(0.00001)
        
    def calculateC2Fit(self,fit,numfit,data,weights):
        tempc2=npy.sum(((fit-data)**2)*weights)
        tempc2=npy.double(tempc2)
        return tempc2/npy.double(len(data)-numfit)
    
    def fitData(self,params,fixes1,constraints,data,weights1=None,verbose=True):
        nparams=len(params)
        npts=len(data)
        fitparams=nparams
        currlambda=self.lambdaval
        fixes=npy.zeros(nparams,dtype=int)
        if(fixes1 is not None):
            for i in range(nparams):
                fitparams-=fixes1[i]
                fixes[i]=fixes1[i]
        weights=npy.ones(npts,dtype=npy.double)
        if(weights1 is not None):
            for i in range(npts): weights[i]=weights1[i]
        chisquared=0.0
        c2old=0.0
        iterations=0
        if(self.maxiter==0):
            fit=self.fitfunc(params)
            outparams=params.copy()
            chisquared=self.calculateC2Fit(fit,fitparams,data,weights)
        else:
            #shiftparams=[params]*(fitparams+1)
            #shiftfit=[self.fitfunc(shiftparams[0])]*(fitparams+1)
            #shiftparams=npy.repeat(params[:,npy.newaxis],axis=1,dtype=npy.double)
            #shiftfit=npy.repeat(self.fitfunc(shiftparams[0])[:,npy.newaxis],axis=1,dtype=npy.double)
            shiftparams=npy.zeros((fitparams+1,len(params)),dtype=npy.double)
            shiftparams[0,:]=params.copy()
            shiftfit=npy.zeros((fitparams+1,npts),dtype=npy.double)
            shiftfit[0,:]=self.fitfunc(shiftparams[0,:])
            dparams=npy.zeros(fitparams,dtype=npy.double)
            chisquared=self.calculateC2Fit(shiftfit[0,:],fitparams,data,weights)
            tempdouble=0.0
            while True:
                c2old=chisquared
                counter=1
                for i in range(nparams):
                    if(fixes[i]==0):
                        shiftparams[counter,:]=shiftparams[0,:]
                        shiftparams[counter,i]+=self.dx
                        counter+=1
                #if(verbose): print(shiftparams)
                for i in range(1,nparams+1):
                    shiftfit[i,:]=self.fitfunc(shiftparams[i,:])
                jacobian=npy.zeros((fitparams,fitparams),dtype=npy.double)
                jvector=npy.zeros(fitparams,dtype=npy.double)
                for i in range(fitparams):
                    for j in range(i+1):
                        jacobian[i,j]=npy.sum(((shiftfit[i+1,:]-shiftfit[0,:])/self.dx)*((shiftfit[j+1,:]-shiftfit[0,:])/self.dx)*weights)
                        if(i!=j): jacobian[j,i]=jacobian[i,j]
                    jvector[i]=npy.sum(((shiftfit[i+1,:]-shiftfit[0,:])/self.dx)*(data-shiftfit[0,:])*weights)
                for k in range(fitparams):
                    jacobian[k,k]*=(1.0+currlambda)
                try:
                    #if(verbose): print(jacobian)
                    dparams=npy.linalg.solve(jacobian,jvector)
                    #if(verbose): print(dparams)
                except:
                    print('singular matrix encountered')
                    break
                counter=0
                for i in range(nparams):
                    if(fixes[i]==0):
                        shiftparams[0,i]+=dparams[counter]
                        if(constraints is not None):
                            shiftparams[0,i]=npy.maximum(shiftparams[0,i],npy.double(constraints[0][i]))
                            shiftparams[0,i]=npy.minimum(shiftparams[0,i],npy.double(constraints[1][i]))
                        counter+=1
                shiftfit[0,:]=self.fitfunc(shiftparams[0,:])
                chisquared=self.calculateC2Fit(shiftfit[0,:],fitparams,data,weights)
                iterations+=1
                if(verbose):
                    print('iteration '+str(iterations)+' c2 = '+str(chisquared))
                if(iterations==self.maxiter):
                    break
                tempdouble=(c2old-chisquared)/chisquared
                #print('f c2 change = '+str(tempdouble))
                if(tempdouble>=0.0 and currlambda>=0.0): currlambda/=10.0
                else: currlambda*=10.0
                if(tempdouble<self.toler and c2old>chisquared): break
            fit=shiftfit[0,:]
            outparams=shiftparams[0,:]
        return iterations,chisquared,outparams,fit
    
if __name__ == "__main__":
    npts=100
    import random
    import gausfunc
    gf=gausfunc.gausfunc()
    def fitfunc(params):
        #function for a single gausian
        #params are 0base,1amp,2xc,3stdev
        #xvals=npy.array((range(npts)),dtype=npy.double)
        #exparr=((xvals-params[2])**2)/(2.0*(params[3]**2))
        #exparr=npy.exp(-exparr)
        #exparr=exparr*params[1]+params[0]
        #return npy.double(exparr)
        func=params[0]+params[1]*gf.get_func(-params[2],npts,1.0,params[3])
        return npy.double(func)
    testdata=fitfunc([1.0,10.0,45.0,2.0])+[random.gauss(0.0,0.5) for i in range(npts)]
    print(testdata)
    constraints=[[0.0,5.0,25.0,0.5],[3.0,20.0,75.0,5.0]]
    baseguess=npy.min(testdata)
    ampguess=npy.max(testdata)-baseguess
    xcguess=npy.argmax(testdata)
    params=[baseguess,ampguess,xcguess,2.0]
    fixes=[0,0,0,0]
    fitclass=NLLSFit(fitfunc)
    print(fitclass.maxiter)
    iterations,c2,params,fit=fitclass.fitData(params,fixes,constraints,testdata)
    print('iterations = '+str(iterations))
    print('c2 = '+str(c2))
    print('params:')
    print(params)
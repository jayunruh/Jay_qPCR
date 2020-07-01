# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:19:56 2020
License: GNU GPLv2
@author: Jay Unruh
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import FitLogistic
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool

#these are the names of our qPCR targets, two for detection and one human for reference
targets=['N1','N2','RPP30']
#     """
#     here we read an analysis_result.json file from newer versions of the quantstudio(tm) software
#     """
#     wellres=edsdict['wellResults']
#     CT=[]
#     sname=[]
#     wellpos=[]
#     wellname=[]
#     target=[]
#     status=[]
#     curves=[]
#     wellnameslist=makeWellNames(16,24,0)
#     for i in range(len(wellres)):
#         welldict=wellres[i]
#         wellposval=welldict['wellIndex']
#         wellpos.append(wellposval)
#         wellname.append(wellnameslist[wellposval])
#         sname.append(welldict['sampleName'])
#         rxntemp=welldict['reactionResults']
#         target.append(rxntemp[0]['targetName'])
#         ampdict=rxntemp[0]['amplificationResult']
#         curves.append(ampdict['deltaRn'])
#         CT.append(ampdict['cq'])
#         status.append(ampdict['ampStatus'])
#     #now convert into dataframes
#     resdf=pd.DataFrame({'Well':wellpos,'Well Position':wellname,'Sample':sname,'Cq':CT,'Target':target,'Amp Status':status})
#     if('Amp Status' in resdf.columns.tolist()):
#         resdf.loc[resdf['Amp Status']=='NO_AMP','Cq']=float('nan')
#     resdf.loc[resdf['Cq']==-1,'Cq']=float('nan')
#     curves=np.array(curves)
#     #curves=curves.T
#     curvedf=pd.DataFrame(columns=wellname)
#     for i in range(curves.shape[0]):
#         #curvedf[curvelabels[i]]=curvearr[i,:]
#         curvedf[wellname[i]]=curves[i,:]
#     return resdf,curvedf

def selectWellNames(wells,direction,maxwells):
    """
    generates a selected well name list from the wells index list (starting from the first well)
    """
    if(maxwells==384):
        wellnames=makeWellNames(16,24,direction)
    else:
        wellnames=makeWellNames(8,12,direction)
    #print(wellnames)
    wellnames=np.array(wellnames)
    wellindices=np.array(wells)
    wellindices=wellindices-wellindices[0]
    selnames=wellnames[wellindices]
    return selnames

def makeWellNames(nrows,ncols,direction):
    """
    makes an array of well names
    direction 0 means row first, 1 means column first
    """
    if(direction==0):
        wellnames=[getWellName(i+1,j+1) for i in range(nrows) for j in range(ncols)]
    else:
        wellnames=[getWellName(i+1,j+1) for j in range(ncols) for i in range(nrows)]
    return wellnames

def well2rowcol(wellname):
    """
    this function takes a wellname (e.g. A10) and returns row number, col number and well letter
    """
    welllett=wellname[0]
    wellnum=wellname[1:len(wellname)]
    return [(ord(welllett)-64),int(wellnum),welllett]

def getWellName(row,col):
    rowlett=chr(row+64)
    return rowlett+str(col)

def getTriple(wellname1):
    rowcol=well2rowcol(wellname1)
    wellname2=getWellName(rowcol[0],rowcol[1]+1)
    wellname3=getWellName(rowcol[0],rowcol[1]+2)
    return [wellname1,wellname2,wellname3]

def aggregate96WellData(resdflist,ampdflist):
    """
    this aggregates up to 4 96 well plates into a 384 well plate
    order is [[1,3],[2,4]]
    only the Well Position and ampdf column labels are translated
    note that the original list is changed by this process
    """
    nplates=len(resdflist)
    if(nplates>4): nplates=4
    def shiftWellName(wellname,rowshift,colshift):
        rowcol=well2rowcol(wellname)
        newrow=rowcol[0]+rowshift
        newcol=rowcol[1]+colshift
        return getWellName(newrow,newcol)
    for i in range(1,nplates):
        colshift=0
        rowshift=8
        if(i==2):
            colshift=12
            rowshift=0
        if(i==3):
            colshift=12
            rowshift=8
        oldwellnames=resdflist[i]['Well Position']
        newwellnames=[shiftWellName(oldwellnames[i],rowshift,colshift) for i in range(len(oldwellnames))]
        resdflist[i]['Well Position']=newwellnames
        oldcollabels=ampdflist[i].columns
        startcol=0
        if(oldcollabels[startcol]!='A1'): 
            startcol=1
        newcollabels=[shiftWellName(oldcollabels[i],rowshift,colshift) for i in range(startcol,len(oldcollabels))]
        if(startcol==1):
            newcollabels.insert(0,oldcollabels[0])
        ampdflist[i].columns=newcollabels
    ampdf=pd.concat(ampdflist,axis=1)
    resdf=pd.concat(resdflist)
    resdf.reset_index(drop=True,inplace=True)
    return resdf,ampdf

def transformResDF(resdf):
    """
    this function takes a results data frame and adds row, col, rowlett columns
    also replaces 'Undetermined' with 'nan' and converts ct to float type
    """
    wellcoordlist=[well2rowcol(resdf['Well Position'][i]) for i in range(len(resdf))]
    resdf['row']=[wellcoordlist[i][0] for i in range(len(wellcoordlist))]
    resdf['col']=[wellcoordlist[i][1] for i in range(len(wellcoordlist))]
    resdf['rowlett']=[wellcoordlist[i][2] for i in range(len(wellcoordlist))]
    resdf.replace('Undetermined','nan',inplace=True)
    resdf.Cq=resdf.Cq.astype(dtype=float,errors='ignore')
    
def pivotAmpDF(ampdf):
    return ampdf.pivot(index='Cycle Number',columns='Well Position',values='dRn')
    
def makeHeatMap(resdf,ctcol,vmin=25,vmax=55):
    """
    this makes a heatmap out of the ct values in the results dataframe
    row and column values must be in 'row' and 'col' columns
    """
    mincol=min(resdf.col)
    maxcol=max(resdf.col)
    minrow=min(resdf.row)
    maxrow=max(resdf.row)
    ctvals=np.zeros((maxrow-minrow+1,maxcol-mincol+1))
    ctvals[resdf.row-1,resdf.col-1]=resdf[ctcol]
    #print(ctvals.shape)
    rowlabels=[chr(i+64) for i in range(minrow,maxrow+1)]
    #print(rowlabels)
    collabels=range(mincol,maxcol+1)
    fig=plt.figure(figsize=(16,8))
    ax=plt.subplot()
    #mask=np.isnan(ctvals)
    mappable=ax.imshow(ctvals,vmin=vmin,vmax=vmax)
    ax.set_yticks(np.arange(len(rowlabels)))
    ax.set_yticklabels(rowlabels)
    ax.set_xticks(np.arange(len(collabels)))
    ax.set_xticklabels(collabels)
    fig.colorbar(mappable)
    plt.show()
    
def makeHoverHeatMap(df,labelcol,hovercols,vmin=25,vmax=45):
    """
    this makes a holoviews heatmap from the dataframe
    x and y and z (column and row and Ct) columns should be the first two columns
    labelcol values are shown on the heatmap
    hovercols values will be shown on hover
    """
    hv.extension('bokeh',width=90)
    heatmap=hv.HeatMap(df)
    #tooltips=[('Row','@Row'),('Col','@Column'),('JCT','@JCT'),('CT','@CT'),('JCategory','@JCategory'),('Category','@Category'),('Sample','@Sample')]
    tooltips=[(hovercols[i],'@'+hovercols[i]) for i in range(len(hovercols))]
    hover=HoverTool(tooltips=tooltips)
    heatmap.opts(tools=[hover], colorbar=True, invert_yaxis=True, width=1500, height=800, clim=(vmin,vmax),xlim=(0.5,24.5))
    return heatmap*hv.Labels(heatmap, vdims=[labelcol]) 

def makeHoverCarpet(carpets,collabels,rowlabels,maxval,extralabels):
    """
    makes a 3 carpet heatmap
    """
    if(len(rowlabels)<1):
        print('no wells to show')
        return
    hv.extension('bokeh',width=90)
    sprowlabels=[rowlabels[i]+','+extralabels[i] for i in range(len(rowlabels))]
    heatmap1=hv.HeatMap((collabels,sprowlabels,carpets[0])).opts(title=targets[0]+' dRn')
    heatmap2=hv.HeatMap((collabels,sprowlabels,carpets[1])).opts(title=targets[1]+' dRn')
    heatmap3=hv.HeatMap((collabels,sprowlabels,carpets[2])).opts(title=targets[2]+' dRn')
    tooltips=[('Well','@y'),('Cycle','@x'),('dRn','@z')]
    hover=HoverTool(tooltips=tooltips)
    layout=heatmap1+heatmap2+heatmap3
    layout.opts(opts.HeatMap(tools=[hover],cmap='jet',colorbar=True, invert_yaxis=True,clim=(0.0,maxval),width=400,height=800,xlabel='Cycle',ylabel='Wells'))
    return layout

def makeHoverScatter(combineddf1,columns,labels,xlims,ylims):
    """
    makes a 3 plot scatter layout
    labels is 3 x nlabels array where first two are x and y others are tooltips
    same with columns
    e.g. ['Amp1','CT1','Category','Sample']
    all x and y values are set to min 0
    """
    hv.extension('bokeh',width=90)
    def setDFColLim(df,column,lim):
        lowind=np.where(df[column]<lim)
        tempvals=df[column].values
        tempvals[lowind]=lim
        df[column]=tempvals
        
    combineddf=combineddf1.copy()
    combineddf.replace(float('NaN'),0.0,inplace=True)
    for i in range(3):
        for j in range(2):
            setDFColLim(combineddf,columns[i][j],0.0)
    tooltips1=[(labels[0][i]+':','@'+columns[0][i]) for i in range(len(labels[0]))]
    hover1=HoverTool(tooltips=tooltips1)
    tooltips2=[(labels[1][i]+':','@'+columns[1][i]) for i in range(len(labels[1]))]
    hover2=HoverTool(tooltips=tooltips2)
    tooltips3=[(labels[2][i]+':','@'+columns[2][i]) for i in range(len(labels[2]))]
    hover3=HoverTool(tooltips=tooltips3)
    scat1=hv.Scatter(combineddf,columns[0][0],vdims=columns[0][1:]).opts(tools=[hover1],size=8,alpha=0.5,width=300,title=targets[0],ylim=ylims,xlim=xlims)
    scat2=hv.Scatter(combineddf,columns[1][0],vdims=columns[1][1:]).opts(tools=[hover2],size=8,alpha=0.5,width=300,title=targets[1],ylim=ylims,xlim=xlims)
    scat3=hv.Scatter(combineddf,columns[2][0],vdims=columns[2][1:]).opts(tools=[hover3],size=8,alpha=0.5,width=300,title=targets[2],ylim=ylims,xlim=xlims)
    layout=scat1+scat2+scat3
    return layout
    
def getPlotGrid(resdf,uampdf,maxamp,logscale):
    """
    this function plots all of the amplification curves on a grid
    """
    fig=plt.figure(figsize=(16,16))
    for i in range(16):
        for j in range(8):
            ax=plt.subplot(16,8,(j+i*8+1))
            wellname1=chr(65+i)+str(j*3+1)
            wellname2=chr(65+i)+str(j*3+2)
            wellname3=chr(65+i)+str(j*3+3)
            if(wellname1 in uampdf.columns):
                plt.plot(uampdf[wellname1])
                plt.plot(uampdf[wellname2])
                plt.plot(uampdf[wellname3])
                ax.set_ylim((0.001,2.0*maxamp))
                ax.set_yscale('log')
                # Turn off tick labels
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ctvals=[getResVal(resdf,wellname1,'Cq'),getResVal(resdf,wellname2,'Cq'),getResVal(resdf,wellname3,'Cq')]
                if(not math.isnan(ctvals[0])): ctvals[0]=int(ctvals[0])
                if(not math.isnan(ctvals[1])): ctvals[1]=int(ctvals[1])
                if(not math.isnan(ctvals[2])): ctvals[2]=int(ctvals[2])
                ax.set_xlabel(wellname1+','+str(ctvals))
                #ax.set_xlabel(wellname1)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    
def getTriplePlot(wellname1,uampdf,ymax,logscale=False):
    """
    plots a well triple (different targets) based on the first well name
    """
    rowcol=well2rowcol(wellname1)
    wellname2=getWellName(rowcol[0],rowcol[1]+1)
    wellname3=getWellName(rowcol[0],rowcol[1]+2)
    fig=plt.figure(figsize=(8,8))
    ax=plt.subplot()
    plt.plot(uampdf[wellname1],label=wellname1)
    plt.plot(uampdf[wellname2],label=wellname2)
    plt.plot(uampdf[wellname3],label=wellname3)
    ax.set_ylim((-0.1,ymax))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Delta Rn')
    if(logscale):
        ax.set_ylim(0.001,ymax)
        ax.set_yscale('log')
    ax.legend()
    plt.show()
    
def getCatCarpet(combineddf,tracesdf,catcolumn,cat,extracolumn):
    """
    here we make carpets out of all of the triples in a category
    """
    rownames=[chr(i+65) for i in range(16)]
    colnames=[]
    for i in range(8):
        triplenames=[str(i*3+1),str(i*3+2),str(i*3+3)]
        colnames.append(triplenames)
    colnames=np.array(colnames)
    wellnamelist=combineddf['WellName'].values.tolist()
    #now show all of the triples in a particular category
    carpet=[]
    catwelllist=[]
    extralist=[]
    for col in colnames[:,0]:
        for row in rownames:
            wellname=row+col
            wellname2=row+str(int(col)+1)
            wellname3=row+str(int(col)+2)
            if(wellname in wellnamelist):
                tabindex=wellnamelist.index(wellname)
                if(combineddf[catcolumn][tabindex]==cat):
                    catwelllist.append(wellname)
                    extralist.append(combineddf[extracolumn][tabindex])
                    curve=tracesdf[wellname].values
                    curve2=tracesdf[wellname2].values
                    curve3=tracesdf[wellname3].values
                    carpet.append([curve,curve2,curve3])
    #need to sort in order of extra column
    if(len(carpet)<1):
        print('there are no '+cat+' wells')
        return None,None,None
    else: 
        print('there are '+str(len(carpet))+' '+cat+' wells')
        sortorder=np.argsort(extralist)
        #print(sortorder)
        extralist=np.array(extralist)[sortorder]
        catwelllist=np.array(catwelllist)[sortorder]
        carpet=np.array(carpet)
        #reshape the carpet to make it easier to plot
        carpet2=np.array([carpet[sortorder,0,:],carpet[sortorder,1,:],carpet[sortorder,2,:]])
        return carpet2,catwelllist,extralist
    
def getResVal(df,wellname,selcol):
    """
    this function gets a result from a results dataframe by wellname and column
    """
    temp=np.argwhere(df['Well Position'].values==wellname).flat
    if(len(temp)>0):
        wellindex=temp[0]
        return df[selcol][wellindex]
    else:
        return float('nan')
    
def testEmpty(df,colname):
    """
    tests if a dataframe column is empty
    """
    test=np.logical_not(np.isnan(df[colname].values))
    return not np.any(test)

def fitDF(uampdf,resdf):
    """
    this function fits a set of delta Rn curves to logistic functions
    outdf rearranges uampdf in resdf order
    """
    #cycle through and get the logistic fits for all data sets that have data
    #at the same time populate new dataframes for output
    #the curve data frame should have a matrix of curve columns in correct order (according to results table)
    #the results data frame will have Well, Well Position, Sample Name, Target Name, CT, row, col, rowlett,base, amp, kd, n, c2, jct
    outdf=pd.DataFrame({'Cycle':uampdf.index})
    xvals=uampdf.index.values
    outresdf=pd.DataFrame({'Well':resdf['Well'],'Well Position':resdf['Well Position'],'Sample Name':resdf['Sample'],'Target Name':resdf['Target']})
    outresdf['CT']=resdf.Cq
    outresdf['row']=resdf.row
    outresdf['col']=resdf.col
    outresdf['rowlett']=resdf.rowlett
    allparams=[]
    allc2=[]
    for i in range(len(resdf)):
        wellname=resdf['Well Position'][i]
        isempty=testEmpty(uampdf,wellname);
        #print(isempty)
        if(not isempty):
            curve=uampdf[wellname].values
            outdf[wellname]=curve
            #xvals=uampdf['Cycle']
            iterations,c2,params,fit=FitLogistic.runFit(xvals,curve,5,55,0.5,0.0)
            allc2.append(c2)
            allparams.append(params)
            print(wellname+','+str(c2)+','+str(params))
        else:
            nanval=float('nan')
            allc2.append(nanval)
            params=[nanval,nanval,nanval,nanval]
            allparams.append(params)
            print(wellname+','+str(nanval)+','+str(params))
    outresdf['c2']=np.array(allc2)
    allparams=np.array(allparams)
    outresdf['baseline']=allparams[:,0]
    outresdf['amp']=allparams[:,1]
    outresdf['c50']=allparams[:,2]
    outresdf['n']=allparams[:,3]
    return outresdf,outdf

def computeJCTValues(outresdf,minamp,threshfrac,maxcntljct,cntlwells,paramslims):
    """
    this function computes JCT values and adds them to the outresdf frame
    also filters based on whether fit values are within limits
    filter order is jct,amp,n
    """
    #need to estimate my modified ct value
    #cntlwells has the list of first wells of the control triples (order target1, target2, target3)
    #assume 2 control wells for now
    #set the threshold at threshfrac of those amplitudes if those fits are good
    #if either set of fits are not good, need to rerun this plate
    c1n=getTriple(cntlwells[0])
    c2n=getTriple(cntlwells[1])
    ctl1amps=np.array([getResVal(outresdf,c1n[0],'amp'),getResVal(outresdf,c1n[1],'amp'),getResVal(outresdf,c1n[2],'amp')])
    ctl2amps=np.array([getResVal(outresdf,c2n[0],'amp'),getResVal(outresdf,c2n[1],'amp'),getResVal(outresdf,c2n[2],'amp')])
    print('control 1 amps '+str(ctl1amps))
    print('control 2 amps '+str(ctl2amps))
    #minamp=1
    ctl1bad=np.any(ctl1amps<minamp)
    ctl2bad=np.any(ctl2amps<minamp)

    try:
        if(ctl1bad or ctl2bad):
            raise RuntimeError('bad control error')
    except:
        print('controls are bad, rerun plate')

    #threshfrac=0.05
    threshvals=threshfrac*0.5*(ctl1amps+ctl2amps)
    print('threshholds '+str(threshvals))

    def getJCT(threshval,base,amp,c50,nval):
        #if the threshold value is above the curve, report the maximum ct value
        if((threshval-base)>=amp): return float('nan')
        return c50-(1.0/nval)*np.log(amp/threshval-1.0)

    def getJCTWell(df,wellname,threshval):
        base=getResVal(df,wellname,'baseline')
        amp=getResVal(df,wellname,'amp')
        c50=getResVal(df,wellname,'c50')
        nval=getResVal(df,wellname,'n')
        return getJCT(threshval,base,amp,c50,nval)
    
    def customFilter(jct,amp,nval,paramslims):
        #filter the paramaters base on parameter limitations
        #returns the JCT value or 'nan'
        #paramslims contains lower and upper limits for
        nanval=float('nan')
        params=[jct,amp,nval]
        for i in range(len(paramslims)):
            if(paramslims[i]):
                if(params[i]<paramslims[i][0]): return nanval
                if(params[i]>paramslims[i][1]): return nanval
        return jct

    #now find the ct values for our controls and check if they are below threshold
    ctl1jct=np.array([getJCTWell(outresdf,c1n[0],threshvals[0]),getJCTWell(outresdf,c1n[1],threshvals[1]),getJCTWell(outresdf,c1n[2],threshvals[2])])
    ctl2jct=np.array([getJCTWell(outresdf,c2n[0],threshvals[0]),getJCTWell(outresdf,c2n[2],threshvals[1]),getJCTWell(outresdf,c2n[2],threshvals[2])])
    print('control 1 jct '+str(ctl1jct))
    print('control 2 jct '+str(ctl2jct))
    #maxcntljct=35.0
    ctl1bad=np.any(ctl1jct>maxcntljct)
    ctl2bad=np.any(ctl2jct>maxcntljct)
    try:
        if(ctl1bad or ctl2bad):
            raise RuntimeError('bad control error')
    except:
        print('controls are bad, rerun plate')

    #ok, now get all of the jct values
    alljct=[]
    for i in range(len(outresdf)):
        wellname=outresdf['Well Position'][i]
        colval=outresdf['col'][i]
        threshindex=(colval-1)%3
        threshval=threshvals[threshindex]
        jct=getJCTWell(outresdf,wellname,threshval)
        amp=getResVal(outresdf,wellname,'amp')
        nval=getResVal(outresdf,wellname,'n')
        jct=customFilter(jct,amp,nval,paramslims)
        alljct.append(jct)
    outresdf['JCT']=np.array(alljct)

def testCT(vals,lim):
    """
    this is the master function that performs hit testing based on lim (ctlim)
    vals is a 3 value array with target1, target2, and hRNA ct values
    values that were undetermined or excluded previously are nan values
    """
    #perform hit testing
    #first check if human rna amplified
    if(math.isnan(vals[2]) or vals[2]>lim): return 'bad'
    #now we know the data isn't bad, check for negative
    if((math.isnan(vals[0]) or vals[0]>lim) and (math.isnan(vals[1]) or vals[1]>lim)): return 'neg'
    #now we know the data isn't negative, check for positive
    if(vals[0]<=lim and vals[1]<=lim): return 'pos'
    #only option left is partial
    return 'par'

def testCtrl(ctrldf,expected,lim):
    """
    this tests whether all of a control plate shows the expected result (passed or failed)
    bad (water control) is special case--all three need to be negative or nan
    """
    res=['passed','passed']
    for i in range(len(ctrldf)):
        ctvals=[ctrldf['CT1'][i],ctrldf['CT2'][i],ctrldf['CT3'][i]]
        jctvals=[ctrldf['JCT1'][i],ctrldf['JCT2'][i],ctrldf['JCT3'][i]]
        if(expected!='bad'):
            if(testCT(ctvals,lim)!=expected): res[0]='failed'
            if(testCT(jctvals,lim)!=expected): res[1]='failed'
        else:
            #for bad, must be all bad:)
            if(ctvals[0]<=lim or ctvals[1]<=lim or ctvals[2]<=lim): res[0]='failed'
            if(jctvals[0]<=lim or jctvals[1]<=lim or jctvals[2]<=lim): res[1]='failed'
    return res
    
def categorizeResults(resdf,ctlim,platename):
    """
    this function collects all of the triples and categorizes them
    categories are bad RNA (hRNA is negative):'bad', negative:'neg', partial positive: 'par', positive: 'pos'
    """
    #rownames=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J','K','L','M','N','O','P']
    rownames=[chr(65+i) for i in range(16)]
    colnames=[]
    for i in range(8):
        triplenames=[str(i*3+1),str(i*3+2),str(i*3+3)]
        colnames.append(triplenames)
    colnames=np.array(colnames)
    col_labels=('Plate_Name','Sample_Name','WellName','Column','Row','JCT1','CT1','JCT2','CT2','JCT3','CT3','JCategory','Category','Amp1','Amp2','Amp3','Base1','Base2','Base3','n1','n2','n3')
    combineddf=pd.DataFrame(columns=col_labels)
    wellnamelist=resdf['Well Position'].values.tolist()
    catlist=np.array(['neg']*len(resdf))
    catlist2=np.array(['neg']*len(resdf))
    
    for i in range(len(colnames)):
        for j in range(len(rownames)):
            wellname=rownames[j]+colnames[i,0]
            wellname2=rownames[j]+colnames[i,1]
            wellname3=rownames[j]+colnames[i,2]
            #print(wellname)
            tabindex=-1
            if(wellname in wellnamelist):
                tabindex=wellnamelist.index(wellname)
                tabindex2=wellnamelist.index(wellname2)
                tabindex3=wellnamelist.index(wellname3)
                jct=resdf['JCT'][tabindex]
                jct2=resdf['JCT'][tabindex2]
                jct3=resdf['JCT'][tabindex3]
                ct=resdf['CT'][tabindex]
                ct2=resdf['CT'][tabindex2]
                ct3=resdf['CT'][tabindex3]
                catlist2[[tabindex,tabindex2,tabindex3]]=testCT([jct,jct2,jct3],ctlim)
                catlist[[tabindex,tabindex2,tabindex3]]=testCT([ct,ct2,ct3],ctlim)
                amps=[resdf['amp'][tabindex],resdf['amp'][tabindex2],resdf['amp'][tabindex3]]
                bases=[resdf['baseline'][tabindex],resdf['baseline'][tabindex2],resdf['baseline'][tabindex3]]
                ns=[resdf['n'][tabindex],resdf['n'][tabindex2],resdf['n'][tabindex3]]
                #now add the values to the combined dataframe
                addvals=[platename,resdf['Sample Name'][tabindex],wellname,colnames[i,0],rownames[j],jct,ct,jct2,ct2,jct3,ct3,catlist2[tabindex],catlist[tabindex]]
                addvals.extend(amps)
                addvals.extend(bases)
                addvals.extend(ns)
                combineddf.loc[len(combineddf)]=addvals
    #print(catlist)
    resdf['JCategory']=catlist2
    resdf['Category']=catlist
    return combineddf

def expandCategorized(combineddf,inclabels,commonlabels,nrepeats):
    """
    this function expands the categorized results back out to a long table
    assume column and row are labeled Column and Row
    """
    #expand the combined results back out to a well by well heatmap
    basecollabels=['Column','Row']
    ninclabels=len(inclabels)
    basecollabels.extend(inclabels)
    basecollabels.extend(commonlabels)
    collabels=[]
    for i in range(nrepeats):
        collabels.append(basecollabels.copy())
        for j in range(ninclabels):
            collabels[i][j+2]=collabels[i][j+2]+str(i+1)
        #print(collabels[i])
    #collabels1=['Column','Row','CT1','JCT1','Category','JCategory','Sample_Name']
    #collabels2=['Column','Row','CT2','JCT2','Category','JCategory','Sample_Name']
    #collabels3=['Column','Row','CT3','JCT3','Category','JCategory','Sample_Name']
    expanddf=pd.DataFrame(columns=tuple(basecollabels))
    for i in range(len(combineddf)):
        for k in range(nrepeats):
            #row=combineddf['Row'][i]
            col=combineddf['Column'][i]
            newrow=[combineddf[collabels[k][j]][i] for j in range(len(basecollabels))]
            newrow[0]=col+k
            expanddf.loc[len(expanddf)]=newrow
    return expanddf

def makeCtrlReport(combineddf,posname,cellname,watername):
    """
    this function makes a  holoviews datatable out of the control wells
    could also add some metrics if desired (pass, fail)
    the control name should be in the Sample_Name column
    typically posname will start with PC, cellname will start with RPE, and watername will be NTC
    """
    pcdf=pd.DataFrame(columns=combineddf.columns)
    celldf=pd.DataFrame(columns=combineddf.columns)
    waterdf=pd.DataFrame(columns=combineddf.columns)
    removelist=[]
    for idx in combineddf.index:
        if(str(combineddf['Sample_Name'][idx]).startswith(posname)):
            pcdf.loc[len(pcdf)]=combineddf.iloc[idx]
            removelist.append(idx)
        if(str(combineddf['Sample_Name'][idx]).startswith(cellname)):
            celldf.loc[len(celldf)]=combineddf.iloc[idx]
            removelist.append(idx)
        if(str(combineddf['Sample_Name'][idx]).startswith(watername)):
            waterdf.loc[len(waterdf)]=combineddf.iloc[idx]
            removelist.append(idx)
    #remove those rows from the combineddf
    combineddf.drop(removelist,inplace=True)
    combineddf.reset_index(inplace=True)
    #now combine the control dataframes, adding a control type column
    pcdf['type']='PC'
    celldf['type']='Cells'
    waterdf['type']='Water'
    ctrldf=pd.concat([pcdf,celldf,waterdf],ignore_index=True)
    #need to drop a bunch of columns for ease of reading
    ctrldf.drop(['Column','Row','Base1','Base2','Base3','n1','n2','n3'],axis=1,inplace=True)
    #now lets make a pass fail table for these
    #pc should all be positive (ct<40), cells only hRNA pos, and water all negative
    ctrlcatdf=pd.DataFrame(columns=('Ctrl_Type','Status','JStatus'))
    pcres=testCtrl(pcdf,'pos',40.0)
    ctrlcatdf.loc[0]=['PC',pcres[0],pcres[1]]
    cellres=testCtrl(celldf,'neg',40.0)
    ctrlcatdf.loc[1]=['Cells',cellres[0],cellres[1]]
    waterres=testCtrl(waterdf,'bad',40.0)
    ctrlcatdf.loc[2]=['Water',waterres[0],waterres[1]]
    return hv.Table(ctrldf).opts(width=900,height=600),hv.Table(ctrlcatdf).opts(width=300,height=300)

def writeResults(dirname,fname,combineddf,results,mapname):
    """
    writes a list of 384 results to a csv file
    fname is renamed to have _calls.csv at the end
    will have Sample_Name, WellName, Result
    results has 4 arrays: bad, neg, par, pos
    """
    if(mapname is not None):
        mapdf=pd.read_excel(mapname,'plate_mapping')
    else:
        nalist=['NA']*len(combineddf)
        nalist=np.array(nalist)
        mapdf=pd.DataFrame({'WellName_384':combineddf['WellName'],'WellName_96':nalist})
    wellnames384=mapdf['WellName_384'].values.tolist()
    prefix=fname[0:-4]
    print(prefix)
    abspath=os.path.abspath(dirname)
    link=os.path.basename(abspath)
    wellnamelist=combineddf['WellName'].tolist()
    outdf=pd.DataFrame(columns=('Specimen ID','Result','Source Well','Data Links','Plate'))
    #first output the bad results
    for well in results[0]:
        if(well in wellnamelist):
            idx=wellnamelist.index(well)
            sname=combineddf['Sample_Name'][idx]
            pname=combineddf['Plate_Name'][idx]
            idx2=wellnames384.index(well)
            outdf.loc[len(outdf)]=[sname,'Inconclusive',mapdf['WellName_96'][idx2],link,pname]
        else:
            print(well+' not found in list')
    for well in results[1]:
        if(well in wellnamelist):
            idx=wellnamelist.index(well)
            sname=combineddf['Sample_Name'][idx]
            pname=combineddf['Plate_Name'][idx]
            idx2=wellnames384.index(well)
            outdf.loc[len(outdf)]=[sname,'Negative',mapdf['WellName_96'][idx2],link,pname]
        else:
            print(well+' not found in list')
    for well in results[2]:
        if(well in wellnamelist):
            idx=wellnamelist.index(well)
            sname=combineddf['Sample_Name'][idx]
            pname=combineddf['Plate_Name'][idx]
            idx2=wellnames384.index(well)
            outdf.loc[len(outdf)]=[sname,'Partial',mapdf['WellName_96'][idx2],link,pname]
        else:
            print(well+' not found in list')
    for well in results[3]:
        if(well in wellnamelist):
            idx=wellnamelist.index(well)
            sname=combineddf['Sample_Name'][idx]
            pname=combineddf['Plate_Name'][idx]
            idx2=wellnames384.index(well)
            outdf.loc[len(outdf)]=[sname,'Positive',mapdf['WellName_96'][idx2],link,pname]
        else:
            print(well+' not found in list')
    outdf['Storage Rack']='RackXXX'
    outdf['Storage Location']='XXX'
    outdf.to_csv(dirname+prefix+'_calls.csv',index=False)
    return
    
def outputCurves(dirname,fname,uampdf,outresdf,combineddf):
    """
    this function outputs the results in the specified directory
    amplification curves end with _curves.csv and results with _results.csv
    """
    prefix=fname[0:-4]
    print(prefix)
    uampdf.to_csv(dirname+prefix+'_curves.csv',index=False)
    combineddf.to_csv(dirname+prefix+'_combined.csv',index=False)
    outresdf.to_csv(dirname+prefix+'_results.csv',index=False)
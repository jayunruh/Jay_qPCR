# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:00:24 2020
License: GNU GPL v2
@author: Jay Unruh
"""

import pandas as pd
import numpy as np
#import os
import json
from zipfile import ZipFile
import xml.etree.ElementTree as ET

def getqPCRData(fpath):
    """
    this function reads header info, results, and amplification data from a qPCR export worksheet
    """
    headdf=pd.read_excel(fpath,'Results',header=None)
    hl=0
    #first find the blank row (reads as NaN)
    while(not str(headdf[0][hl]).isnumeric()):
        hl+=1
    #print(hl)
    resdf=pd.read_excel(fpath,'Results',skiprows=hl-1)
    header={headdf[0][i]:headdf[1][i] for i in range(hl-1)}
    #can presume that the amplification worksheet has the same header
    ampdf=pd.read_excel(fpath,'Amplification Data',skiprows=hl-1)
    return header,resdf,ampdf

def getEDSData(fpath):
    isjson=False
    with ZipFile(fpath,'r') as zipobj:
        flist=zipobj.namelist()
        if('summary.json' in flist):
            isjson=True
            print('file has json data')
    if(isjson):
        temp=getEDSJSONData(fpath)
    else:
        temp=getEDSData2(fpath)
    return temp

def getEDSData2(fpath):
    """
    parses an EDS file via zip utility
    """
    lines=[]
    meta={}
    resdf=None
    curvedf=None
    with ZipFile(fpath,'r') as zipobj:
        flist=zipobj.namelist()
        for fname in flist:
            if(fname.endswith('analysis_result.txt')):
                with(zipobj.open(fname)) as fp:
                    line=fp.readline().decode("utf-8")
                    while line:
                        lines.append(line[0:-1])
                        line=fp.readline().decode("utf-8")
                #print(lines[0])
                #print(lines[1])
                #print(lines[2])
                resdf,curvedf=parseEDSResults(lines)
            if(fname.endswith('plate_setup.xml')):
                with(zipobj.open(fname)) as fp:
                    tree=ET.parse(fp)
                    root=tree.getroot()
                    meta.update(getMeta(root,['Barcode','Comment']))
            if(fname.endswith('experiment.xml')):
                with(zipobj.open(fname)) as fp:
                    tree=ET.parse(fp)
                    root=tree.getroot()
                    meta.update(getMeta(root,['Name','Description']))
    return resdf,curvedf,meta

def getEDSJSONData(fpath):
    """
    parses an EDS file that keeps it's values in zipped JSON files
    """
    meta={}
    resdf=None
    curvedf=None
    with ZipFile(fpath,'r') as zipobj:
        flist=zipobj.namelist()
        for fname in flist:
            if(fname.endswith('analysis_result.json')):
                with(zipobj.open(fname)) as fp:
                    resdict=json.load(fp)
                resdf,curvedf=parseEDSJSONResults(resdict)
            if(fname=='summary.json'):
                with(zipobj.open(fname)) as fp:
                    sumdict=json.load(fp)
                    meta.update(getJSONMeta(sumdict,['name','description']))
                    meta['Name']=meta.pop('name')
                    meta['Description']=meta.pop('description')
                    #print(meta['Description'])
            if(fname.endswith('plate_setup.json')):
                with(zipobj.open(fname)) as fp:
                    psdict=json.load(fp)
                    meta.update(getJSONMeta(psdict,['plateBarcode','comment']))
                    meta['Barcode']=meta.pop('plateBarcode')
                    meta['Comment']=meta.pop('comment')
    return resdf,curvedf,meta

def getMeta(root,keys):
    """
    finds keys in xml tree (first branch) and returns a dictionary
    """
    meta={}
    for i in range(len(keys)):
        try:
            meta[keys[i]]=root.find(keys[i]).text
        except:
            meta[keys[i]]='NA'
    return meta

def getJSONMeta(jsondict,keys):
    """
    find keys in json dictionary
    """
    meta={}
    for i in range(len(keys)):
        try:
           meta[keys[i]]=jsondict[keys[i]]
        except:
           meta[keys[i]]='NA'
    return meta
                        
def parseEDSResults(lines):
    """
    parses the 'analysis_results.txt' from the eds file
    """
    pos=0
    while(not lines[pos].startswith('Well')):
        pos+=1
    collabels=lines[pos].split('\t')
    resarr=[]
    curvearr=[]
    for i in range(pos+1,len(lines)-2,3):
        resarr.append(lines[i].split('\t')[0:-1])
        curvearr.append(lines[i+2].split('\t')[1:-1])
    resarr=np.array(resarr)
    curvearr=np.array(curvearr)
    resdf=pd.DataFrame(columns=tuple(collabels))
    for i in range(resarr.shape[1]):
        resdf[collabels[i]]=resarr[:,i]
    wellnum=resdf['Well'].astype(int)
    maxwell=max(wellnum)
    nwells=96
    if(maxwell>nwells): nwells=384
    wellnames=selectWellNames(wellnum,0,nwells)
    resdf['Well Position']=wellnames
    resdf.rename(columns={"Sample Name": "Sample","Ct":"Cq","Avg Ct":"Avg Cq","Ct SD":"Cq SD","Delta Ct":"Delta Cq","Detector":"Target"},inplace=True)
    #for i in range(len(resdf)):
    #    if('Amp Status' in resdf.columns.tolist()):
    #        if(float(resdf['Amp Status'][i])<0):
    #            resdf['Cq'][i]=float('nan')
    if('Amp Status' in resdf.columns.tolist()):
        resdf.loc[resdf['Amp Status']<0,'Cq']=float('nan')
    resdf.loc[resdf['Cq']==45,'Cq']=float('nan')
    #now shift the well position column just after the Well column
    collabels=resdf.columns.tolist()
    newcollabels=[collabels[0],collabels[-1]]
    newcollabels.extend(collabels[1:-2])
    resdf=resdf[newcollabels]
    curvedf=pd.DataFrame(columns=wellnames)
    print(curvearr.shape)
    for i in range(curvearr.shape[0]):
        #curvedf[curvelabels[i]]=curvearr[i,:]
        curvedf[wellnames[i]]=curvearr[i,:]
    curvedf=curvedf.apply(pd.to_numeric,errors='coerce')
    return resdf,curvedf

def parseEDSJSONResults(edsdict):
    """
    here we read an analysis_result.json file from newer versions of the quantstudio(tm) software
    """
    wellres=edsdict['wellResults']
    CT=[]
    sname=[]
    wellpos=[]
    wellname=[]
    target=[]
    status=[]
    curves=[]
    wellnameslist=makeWellNames(16,24,0)
    for i in range(len(wellres)):
        welldict=wellres[i]
        wellposval=welldict['wellIndex']
        wellpos.append(wellposval)
        wellname.append(wellnameslist[wellposval])
        sname.append(welldict['sampleName'])
        rxntemp=welldict['reactionResults']
        target.append(rxntemp[0]['targetName'])
        ampdict=rxntemp[0]['amplificationResult']
        curves.append(ampdict['deltaRn'])
        CT.append(ampdict['cq'])
        status.append(ampdict['ampStatus'])
    #now convert into dataframes
    resdf=pd.DataFrame({'Well':wellpos,'Well Position':wellname,'Sample':sname,'Cq':CT,'Target':target,'Amp Status':status})
    if('Amp Status' in resdf.columns.tolist()):
        resdf.loc[resdf['Amp Status']=='NO_AMP','Cq']=float('nan')
    resdf.loc[resdf['Cq']==-1,'Cq']=float('nan')
    curves=np.array(curves)
    #curves=curves.T
    curvedf=pd.DataFrame(columns=wellname)
    for i in range(curves.shape[0]):
        #curvedf[curvelabels[i]]=curvearr[i,:]
        curvedf[wellname[i]]=curves[i,:]
    return resdf,curvedf

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
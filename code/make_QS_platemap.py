# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:10:02 2020
License: GNU GPLv2
this is a command line tool to convert sample names into a QS plate setup file
@author: Jay Unruh
"""

import pandas as pd
import numpy as np
import sys

def readSSFile(ssfile):
    #need to read the header information (separated by blank row from setup info)
    f=open(ssfile,'r')
    header=[]
    try:
        temp=f.readline()
        pos=0
        while(len(temp.strip())>1):
            header.append(temp)
            temp=f.readline()
            pos+=1
        blankpos=pos
        temp=f.readline()
        pos+=1
        if(temp.startswith('[')): blankpos=pos
        while(f.readline()):
            pos+=1
    finally:
        f.close()
    resdf=pd.read_csv(ssfile,sep='\t',skiprows=blankpos+1,nrows=(pos-blankpos-1),thousands=',')
    return header,resdf

def amendSS(ssdf,mapdf,specimendf):
    #ok, now remake the ssdf, using the mapdf to map 394 well names to 96 well names and specimens
    #anything with a map to NTC, PC, or 00 shouldn't be mapped
    wellnames384=mapdf['WellName_384'].values.tolist()
    wellnames96=specimendf['WellName'].values.tolist()
    sampnames=[]
    for i in range(len(ssdf)):
        name384=ssdf['Well Position'][i]
        mapindex=wellnames384.index(name384)
        name96=mapdf['WellName_96'][mapindex]
        if(name96 in wellnames96):
            sindex=wellnames96.index(name96)
            sid=specimendf['Specimen_ID'][sindex]
            sampnames.append(sid)
            #ssdf['Sample Name'][i]=sid
        else:
            #if(name96=='00'): ssdf['Sample Name'][i]=''
            #else: ssdf['Sample Name'][i]=name96
            if(name96=='00'): sampnames.append('')
            else: sampnames.append(name96)
        print([name384,name96,sampnames[i]])
    ssdf['Sample Name']=np.array(sampnames)
    
def writeSSFile(fname,header,ssdf):
    f=open(fname,'w')
    for line in header:
        f.write(line)
    f.write('\n')
    f.write('[Sample Setup]\n')
    ssdf.to_csv(f,index=False,sep='\t',line_terminator='\n')
    
def replaceHeadValue(head,key,value):
    for i in range(len(head)):
        if(head[i].startswith(key)):
            head[i]=key+' = '+value+'\n'
    return
    
if __name__ == "__main__":
    if(len(sys.argv)>1):
        ssname=sys.argv[1]
        mapname=sys.argv[2]
        specname=sys.argv[3]
        outname=sys.argv[4]
    else:
        #print('wrong format')
        print('missing expected arguments')
        print('expected usage: python make_QS_platemap.py template.txt plate_translation.xlsx specimen_id.xls output.txt')
    if(ssname is not None):
        specdf=pd.read_excel(specname)
        platename=specdf['SRCRackID'][0]
        platedate=specdf['Time'][0]
        specdf=specdf.drop(0)
        specdf.reset_index(inplace=True)
        specdf.astype({'Position':'int'})
        specdf.sort_values('Position',inplace=True)
        rownames=[chr(65+i) for i in range(8)]
        colnames=[str(i+1) for i in range(12)]
        print(rownames)
        print(colnames)
        wellnames=[str(rownames[i])+str(colnames[j]) for j in range(len(colnames)) for i in range(len(rownames))]
        specimendf=pd.DataFrame({'WellName':wellnames,'Specimen_ID':specdf['SRCTubeID']})
        sshead,ssdf=readSSFile(ssname)
        replaceHeadValue(sshead,'* Experiment Comment',platename)
        mapdf=pd.read_excel(mapname,'plate_mapping')
        amendSS(ssdf,mapdf,specimendf)
        #could put data in Biogroup Name or Comments if desired
        #ssdf['Biogroup Name']=np.array([platename]*len(ssdf))
        #ssdf['Comments']=np.array([platedate]*len(ssdf))
        writeSSFile(outname,sshead,ssdf)
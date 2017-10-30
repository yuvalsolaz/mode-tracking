
import pandas as pd
import numpy as np

from detect_peaks import detect_peaks
from pandas import rolling_median

from consts import *

## Functions
def RollingPercentile(values,a):
    res = np.percentile(values,a)
    return res

def RollingEntropy(values):
    res = (-values*np.log2(values)).sum(axis=0)
    return res

def RollingEnergy(values):
    N = len(values)
    res = np.sum(np.abs(values) ** 2) / N
#    res = (-values*np.log2(values)).sum(axis=0)
    return res

def NormValues(values):
    Res = values/values.max()
    return Res

def LowPassFilter(values):
    threshold = values.mean()+3*values.std()
    ResE = rolling_median(values, window=15, center=True).fillna(method='bfill').fillna(method='ffill')
    #ResE = NormValues(ResEtemp)
    return ResE

def RemoveOutlier (values):
    threshold = values.mean()+3*values.std()
    ResEtemp = rolling_median(values, window=15, center=True).fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(values - ResEtemp)
    outlier_idx = difference > threshold
    #outlier_idx = values > threshold
    values[outlier_idx] = threshold
    ##ResE = NormValues(ResEtemp)
    return values

 # Average absoulute difference
def AveAbsDiff(values):
    res = np.mean(np.abs(values - values.mean()))
    return res

# Signal Magnitude Area
def SigMagArea(values):
    res = np.sum(values.mean())
    return res

# Zero crossing rate
def ZeroCorsRate(values):
    res = (np.nonzero(np.diff(values>0))[0]).size
    return res

# g-crossing rate
def GCorsRate(values):
    res = (np.nonzero(np.diff(values>1))[0]).size
    return res

# medain-crossing rate
def MCorsRate(values):
    tem = np.median(values)
    res = (np.nonzero(np.diff(values>tem))[0]).size
    return res

## peaks detection :
def peaks(values):
    mph = 1.1 #0 ## minimum peak height
    mpd = 5 ## minimum peak distance = 20
    return len(detect_peaks(values, mph, mpd, show=False))

# rolling window on 2d numpy array. return crrelation between first and second array colomn
# input 2d array with 2 colomns and rolling window size
# output: 1d array with correlation results

def window_correlation(a,window_size):
    ## pad array start with zeros for rolling window compatibility
    c = np.append(np.zeros((window_size,2),np.float64 ), a , axis=0)

    ## stacks windows :
    interlaceing = np.hstack(c[i:1+i-window_size or None:1] for i in range(0,window_size) )

    # get left and right indices :
    l = interlaceing[:,range(0, 2*window_size, 2)]
    r = interlaceing[:,range(1, 2*window_size, 2)]

    ## correlates left and right values
    return  np.array([np.corrcoef(l[i] , r[i],rowvar=0)[0,1] for i in range(len(l)-1)])

    # test method :
    if 0 != 0 :
       v = np.linspace(0.0,20.0,20).reshape(10,2)
       print ('v= ' , v.shape ,v)
       print (window_correlation(v,4))


def addFeatures(idf,g = 1.0):

    gfx = idf['gfx'] / g
    gfy = idf['gFy'] / g
    gfz = idf['gFz'] / g

    # remove outliers
    rawgyro = np.sqrt(idf['wx']**2 + idf['wy']**2 + idf['wz']**2)
    idf['gyro'] =RemoveOutlier(rawgyro)
    rawforce = np.sqrt(gfx**2 + gfy**2 + gfz**2)
    idf['gforce'] = RemoveOutlier(rawforce)

    #df['gforce'] = np.sqrt(df['gfx']**2 + df['gFy']**2 + df['gFz']**2)
    #df['gyro'] = np.sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)
    ## calculates statistics features on rolling window :

    idf['agforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['agyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).mean()

    idf['mgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['mgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).median()

    idf['vgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['vgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()

    idf['maxgforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxgyro']   = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).max()

    idf['maxgforceabs'] = abs(idf['maxgforce'])
    idf['maxgyroabs']   = abs(idf['maxgyro'])

    idf['mingforce'] = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['mingyro']  = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1 ,center=False).min()

    idf['mingforceabs'] = abs(idf['mingforce'])
    idf['mingyroabs']   = abs(idf['mingyro'])

    idf['ampgforce'] = idf['maxgforce'] - idf['mingforce']
    idf['ampgyro']  = idf['maxgyro'] - idf['mingyro']

    idf['amAccGyro'] = idf['ampgforce']*idf['ampgyro']

    roll_force = idf['gforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_gyro = idf['gyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    ## iqr calculations:

    q25 = roll_force.apply(RollingPercentile, [25.0])
    q75 = roll_force.apply(RollingPercentile, [75.0])
    q25 = roll_gyro.apply(RollingPercentile, [25.0])
    q75 = roll_gyro.apply(RollingPercentile, [75.0])
    idf['iqrforce'] = q75 - q25
    idf['iqqgyro'] = q75 - q25



    ## entropy calculations:
    idf['entforce'] = roll_force.apply(RollingEntropy);
    idf['entgyro'] = roll_gyro.apply(RollingEntropy);


    ## energy calculations:
    idf['enyforce'] = roll_force.apply(RollingEnergy);
    idf['enygyro'] = roll_gyro.apply(RollingEnergy);

    ## ratio calculations
    idf['MultiGyroAcc'] = idf['maxgforce']*idf['maxgyro']

    idf['MultiVarGyroAcc']= idf['vgforce']*idf['vgyro']


    idf['skforce'] = roll_force.skew()
    idf['skgyro'] = roll_gyro.skew()

    idf['kuforce'] = roll_force.kurt()
    idf['kugyro'] = roll_gyro.kurt()

    # Average absoulute difference
    idf['aadforce'] = roll_force.apply(AveAbsDiff)
    idf['aadgyro'] = roll_gyro.apply(AveAbsDiff)
    # Signal Magnitude Area
    idf['smaforce'] = roll_force.apply(SigMagArea)
    idf['smagyro'] = roll_gyro.apply(SigMagArea)
    # Zero crossing rate
    idf['zcrforce'] = roll_force.apply(ZeroCorsRate)
    idf['zcrgyro'] = roll_gyro.apply(ZeroCorsRate)


    # new features 22.10.17
    idf['gcrgforce'] = roll_force.apply(GCorsRate)
    idf['mcrgforce'] = roll_force.apply(MCorsRate)
    idf['mcrgyro'] = roll_force.apply(MCorsRate)



    idf['light'] = idf['I'] if 'I' in idf else np.nan
    roll_light = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    idf['aadlight'] = roll_light.apply(AveAbsDiff)
    idf['smalight'] = roll_light.apply(SigMagArea)
    idf['zcrlight'] = roll_light.apply(ZeroCorsRate)



    idf['alight'] = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['mlight'] = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['vlight'] = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['maxlight'] = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['minlight'] = idf['light'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['amplight']  = idf['maxlight'] - idf['minlight']
    idf['enylight'] = roll_light.apply(RollingEnergy);
    q25 = roll_light.apply(RollingPercentile, [25.0])
    q75 = roll_light.apply(RollingPercentile, [75.0])
    idf['iqrlight'] = q75 - q25
    idf['MultiGyroLight'] = idf['maxlight']*idf['maxgyro']
    idf['MultiLightAcc'] = idf['maxgforce']*idf['maxlight']
    idf['MultiVarGyroAcc']= idf['vgforce']*idf['vgyro']
    idf['MultiVarGyroLight']= idf['vlight']*idf['vgyro']
    idf['MultiVarLightAcc']= idf['vgforce']*idf['vlight']
    idf['amAccLight'] = idf['ampgforce']*idf['amplight']
    idf['amLightGyro'] = idf['amplight']*idf['ampgyro']
    idf['peakslight'] = idf['light'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    idf['mcrlight'] = roll_light.apply(MCorsRate)


    idf['peaksgforce'] = idf['gforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peaksgyro'] = idf['gyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    ## gforce gyro light correlation :
    idf['gforcegyrocorr'] = window_correlation(idf[['gforce','gyro']].values,WINDOW_SIZE)

    idf['fxforce'] = RemoveOutlier(gfx)
    idf['fyforce'] = RemoveOutlier(gfy)
    idf['fzforce'] = RemoveOutlier(gfz)

    idf['afxforce'] = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['afyforce'] = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['afzforce'] = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).mean()
    idf['mfxforce'] = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['mfyforce'] = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['mfzforce'] = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).median()
    idf['vfxforce'] = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['vfyforce'] = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['vfzforce'] = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).var()
    idf['maxfxforce'] = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxfyforce'] = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxfzforce'] = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['minfxforce'] = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['minfyforce'] = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['minfzforce'] = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['ampfxforce'] = idf['maxfxforce'] - idf['minfxforce']
    idf['ampfyforce'] = idf['maxfyforce'] - idf['minfyforce']
    idf['ampfzforce'] = idf['maxfzforce'] - idf['minfzforce']
    idf['ampGyrofxforce'] = idf['ampfxforce']*idf['ampgyro']
    idf['ampGyrofyforce'] = idf['ampfyforce']*idf['ampgyro']
    idf['ampGyrofzforce'] = idf['ampfzforce']*idf['ampgyro']

    roll_fxforce = idf['fxforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_fyforce = idf['fyforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_fzforce = idf['fzforce'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)

    idf['enyfxforce'] = roll_fxforce.apply(RollingEnergy)
    idf['enyfyforce'] = roll_fyforce.apply(RollingEnergy)
    idf['enyfzforce'] = roll_fzforce.apply(RollingEnergy)

    idf['aadfxforce'] = roll_fxforce.apply(AveAbsDiff)
    idf['aadfyforce'] = roll_fyforce.apply(AveAbsDiff)
    idf['aadfzforce'] = roll_fzforce.apply(AveAbsDiff)

    idf['smafxforce'] = roll_fxforce.apply(SigMagArea)
    idf['smafyforce'] = roll_fyforce.apply(SigMagArea)
    idf['smafzforce'] = roll_fzforce.apply(SigMagArea)

    idf['gcrfxforce'] = roll_fxforce.apply(GCorsRate)
    idf['gcrfyforce'] = roll_fyforce.apply(GCorsRate)
    idf['gcrfzforce'] = roll_fzforce.apply(GCorsRate)

    idf['mcrfxforce'] = roll_fxforce.apply(MCorsRate)
    idf['mcrfyforce'] = roll_fyforce.apply(MCorsRate)
    idf['mcrfzforce'] = roll_fzforce.apply(MCorsRate)


    idf['peakfxforce'] = idf['fxforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peakfyforce'] = idf['fyforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peakfzforce'] = idf['fzforce'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    ## gyro comp
    idf['wxgyro'] = RemoveOutlier(idf['wx'])
    idf['wygyro'] = RemoveOutlier(idf['wy'])
    idf['wzgyro'] = RemoveOutlier(idf['wz'])
    roll_wxgyro = idf['wxgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_wygyro = idf['wygyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)
    roll_wzgyro = idf['wzgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False)


    idf['enywxgyro'] = roll_wxgyro.apply(RollingEnergy)
    idf['enywygyro'] = roll_wygyro.apply(RollingEnergy)
    idf['enywzgyro'] = roll_wzgyro.apply(RollingEnergy)

    idf['aadwxgyro'] = roll_wxgyro.apply(AveAbsDiff)
    idf['aadwygyro'] = roll_wygyro.apply(AveAbsDiff)
    idf['aadwzgyro'] = roll_wzgyro.apply(AveAbsDiff)

    idf['smawxgyro'] = roll_wxgyro.apply(SigMagArea)
    idf['smawygyro'] = roll_wygyro.apply(SigMagArea)
    idf['smawzgyro'] = roll_wzgyro.apply(SigMagArea)

    idf['mcrwxgyro'] = roll_wxgyro.apply(MCorsRate)
    idf['mcrwygyro'] = roll_wygyro.apply(MCorsRate)
    idf['mcrwzgyro'] = roll_wzgyro.apply(MCorsRate)

    idf['peakwxgyro'] = idf['wxgyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peakwygyro'] = idf['wygyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)
    idf['peakwzgyro'] = idf['wzgyro'].rolling(window=PEAKS_WINDOW_SIZE,min_periods=1 ,center=False).apply(peaks)

    idf['maxwxgyro'] = idf['wxgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxwygyro'] = idf['wygyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['maxwzgyro'] = idf['wzgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).max()
    idf['minwxgyro'] = idf['wxgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['minwygyro'] = idf['wygyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()
    idf['minwzgyro'] = idf['wzgyro'].rolling(window=WINDOW_SIZE,min_periods=1,center=False).min()

    idf['ampwxgyro'] = idf['maxwxgyro'] - idf['minwxgyro']
    idf['ampwygyro'] = idf['maxwygyro'] - idf['minwygyro']
    idf['ampwzgyro'] = idf['maxwzgyro'] - idf['minwzgyro']
    idf['ampGyrowxforce'] = idf['ampwxgyro']*idf['ampgforce']
    idf['ampGyrowyforce'] = idf['ampwygyro']*idf['ampgforce']
    idf['ampGyrowzforce'] = idf['ampwzgyro']*idf['ampgforce']

    idf.fillna(method='ffill', axis=0, inplace=True)
    return idf

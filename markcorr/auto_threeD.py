import numpy as np 
from scipy.stats import rankdata
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import gundam as gun

def comoving_distance_Mpch(redshift, cosmologyH0Om0):
    H0, Om0 = cosmologyH0Om0
    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    littleH = H0/100.
    comDist = cosmology.comoving_distance(redshift).value*littleH
    return comDist

def xi_s(raReal, decReal, zReal, raRand, decRand, zRand, sMin, sNBins, sBinWidth, cosmologyH0Om0,
          doBoot=False):

    distReal = comoving_distance_Mpch(zReal, cosmologyH0Om0)
    distRand = comoving_distance_Mpch(zRand, cosmologyH0Om0)

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal, distReal], names=('ra', 'dec', 'z', 'dcom'))
    rans = Table([raRand, decRand, zRand, distRand], names=('ra', 'dec', 'z', 'dcom'))

    par = gun.packpars(kind='rcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nseps=int(sNBins), 
                       sepsmin=sMin, dseps=sBinWidth, calcdist=False, logseps=True, 
                       estimator='LS', doboot=doBoot) 

    gals['wei'] = 1.
    rans['wei'] = 1.

    result = gun.rcf(gals, rans, par)
    s = result['sm']
    xi = result['xis']

    if doBoot:
        xiErr = result['xiserr']
        return s, xi, xiErr
    else:
        return s, xi

def weighted_xi_s(raReal, decReal, zReal, weightReal, raRand, decRand, zRand, sMin, sNBins, 
                  sBinWidth, cosmologyH0Om0, doBoot=False):

    distReal = comoving_distance_Mpch(zReal, cosmologyH0Om0)
    distRand = comoving_distance_Mpch(zRand, cosmologyH0Om0)

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal, distReal], names=('ra', 'dec', 'z', 'dcom'))
    rans = Table([raRand, decRand, zRand, distRand], names=('ra', 'dec', 'z', 'dcom'))

    par = gun.packpars(kind='rcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nseps=int(sNBins), sepsmin=sMin, dseps=sBinWidth, calcdist=False, logseps=True, estimator='LS', doboot=doBoot) 

    gals['wei'] = weightReal/np.mean(weightReal) # gundam does not normalize the weight inside it. 
    rans['wei'] = 1.

    result = gun.rcf(gals, rans, par)
    s = result['sm']
    weightedXi = result['xis']

    if doBoot:
        weightedXiErr = result['xiserr']
        return s, weightedXi, weightedXiErr
    else:
        return s, weightedXi

def mcf_s(s, xiS, weightedXi):
    Ms = (1 + weightedXi)/(1 + xiS)
    return Ms

def compute_cf(realTab, realProperties, randTab, sMin, sNBins, sBinWidth, doRanking, 
               realRaCol='RA', realDecCol='Dec', realZCol='Z', randRaCol='RA', randDecCol='Dec', 
               randZCol='Z', cosmologyH0Om0=[70.0, 0.3], doBoot=False):

    raReal = realTab[realRaCol]
    decReal = realTab[realDecCol]
    zReal = realTab[realZCol]

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]
    zRand = randTab[randZCol]

    s, xi = xi_s(raReal, decReal, zReal, raRand, decRand, zRand, sMin, sNBins, sBinWidth, cosmologyH0Om0, doBoot=doBoot)  

    sXiMcfs = np.empty((len(s), 0))

    sXiMcfs = np.hstack((sXiMcfs, s.reshape(len(s), 1)))
    sXiMcfs = np.hstack((sXiMcfs, xi.reshape(len(s), 1)))

    if(len(realProperties) >= 1):

        for prop_i in realProperties:
        
            propNow = np.array(realTab[prop_i])

            if doRanking:
                propNowRanked = rankdata(propNow)
                weightReal = propNowRanked
            else:
                weightReal = propNow
                
            s, weightedXi = weighted_xi_s(raReal, decReal, zReal, weightReal, raRand, decRand, 
                                          zRand, sMin, sNBins, sBinWidth, cosmologyH0Om0, 
                                          doBoot=doBoot)

            MsArray = np.array(mcf_s(s, xi, weightedXi)).reshape(len(s), 1)
                    
            sXiMcfs = np.hstack((sXiMcfs, MsArray))
        
    return sXiMcfs
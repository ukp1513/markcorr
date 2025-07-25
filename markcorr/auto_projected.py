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

def omegap_rp(raReal, decReal, zReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, cosmologyH0Om0, doBoot=False):

    distReal = comoving_distance_Mpch(zReal, cosmologyH0Om0)
    distRand = comoving_distance_Mpch(zRand, cosmologyH0Om0)

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal, distReal], names=('ra', 'dec', 'z', 'dcom'))
    rans = Table([raRand, decRand, zRand, distRand], names=('ra', 'dec', 'z', 'dcom'))

    par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=int(rpNBins), seppmin=rpMin, dsepp=rpBinWidth, calcdist=False, logsepp=True, nsepv=int(piNBins),
                       dsepv=piBinWidth, estimator='LS', doboot=False)

    gals['wei'] = 1.
    rans['wei'] = 1.

    result = gun.pcf(gals, rans, par, write=False)
    rp = result['rpm']
    omegaP = result['wrp']
    omegaPErr = result['wrperr'] if doBoot else None

    return rp, omegaP, omegaPErr

def weighted_omegap_rp(raReal, decReal, zReal, weightReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, cosmologyH0Om0, doBoot=False):

    distReal = comoving_distance_Mpch(zReal, cosmologyH0Om0)
    distRand = comoving_distance_Mpch(zRand, cosmologyH0Om0)

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal, distReal], names=('ra', 'dec', 'z', 'dcom'))
    rans = Table([raRand, decRand, zRand, distRand], names=('ra', 'dec', 'z', 'dcom'))

    par = gun.packpars(kind='pcf', h0=H0, omegam=OmegaM, omegal=(1-OmegaM), nsepp=int(rpNBins), seppmin=rpMin, dsepp=rpBinWidth, calcdist=False, logsepp=True, nsepv=int(piNBins),
                       dsepv=piBinWidth, estimator='LS', doboot=False)

    gals['wei'] = weightReal/np.mean(weightReal) # gundam does not normalize the weight inside it.
    rans['wei'] = 1.

    result = gun.pcf(gals, rans, par, write=False)
    rp = result['rpm']
    weightedOmegaP = result['wrp']
    weightedOmegaErr = result['wrperr'] if doBoot else None

    return rp, weightedOmegaP, weightedOmegaErr

def mcf_rp(rp, omegaP, weightedOmegaP):
    MpRp = (1 + (weightedOmegaP/rp))/(1 + (omegaP/rp))
    return MpRp

def do_compute(realTab, realProperties, randTab, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, doRanking, realRaCol='RA', realDecCol='Dec', realZCol='Z', randRaCol='RA', randDecCol='Dec', randZCol='Z', cosmologyH0Om0=None, doBoot=False):

    if cosmologyH0Om0 is None:
        cosmologyH0Om0=[70.0, 0.3]

    raReal = realTab[realRaCol]
    decReal = realTab[realDecCol]
    zReal = realTab[realZCol]

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]
    zRand = randTab[randZCol]

    rp, omegaP, _ = omegap_rp(raReal, decReal, zReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, cosmologyH0Om0, doBoot=doBoot)

    rpOmegaMcfs = np.empty((len(rp), 0))

    rpOmegaMcfs = np.hstack((rpOmegaMcfs, rp.reshape(len(rp), 1)))
    rpOmegaMcfs = np.hstack((rpOmegaMcfs, omegaP.reshape(len(rp), 1)))

    if len(realProperties) >= 1:

        for prop_i in realProperties:

            propNow = np.array(realTab[prop_i])

            if doRanking:
                weightReal = rankdata(propNow)
            else:
                weightReal = propNow

            rp, weightedOmega, _ = weighted_omegap_rp(raReal, decReal, zReal, weightReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, cosmologyH0Om0, doBoot=doBoot)

            MpRpArray = np.array(mcf_rp(rp, omegaP, weightedOmega)).reshape(len(rp), 1)

            rpOmegaMcfs = np.hstack((rpOmegaMcfs, MpRpArray))

    return rpOmegaMcfs

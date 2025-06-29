import numpy as np
from scipy.stats import rankdata
from astropy.table import Table
import gundam as gun

def omega_theta(raReal, decReal, raRand, decRand, thMin, thNBins, thBinWidth, doBoot=False):

    gals = Table([raReal, decReal], names=('ra', 'dec'))
    rans = Table([raRand, decRand], names=('ra', 'dec'))

    par = gun.packpars(kind='acf', nsept=int(thNBins), septmin=thMin, dsept=thBinWidth, logsept=True, estimator='LS', doboot=doBoot)

    gals['wei'] = 1.
    rans['wei'] = 1.

    result = gun.acf(gals, rans, par)
    th = result['thm']
    omega = result['wth']
    omegaErr = result['wtherr'] if doBoot else None

    return th, omega, omegaErr

def weighted_omega_theta(raReal, decReal, weightReal, raRand, decRand, thMin, thNBins, thBinWidth, doBoot=False):

    gals = Table([raReal, decReal], names=('ra', 'dec'))
    rans = Table([raRand, decRand], names=('ra', 'dec'))

    par = gun.packpars(kind='acf', nsept=int(thNBins), septmin=thMin, dsept=thBinWidth, logsept=True, estimator='LS', doboot=doBoot)

    gals['wei'] = weightReal/np.mean(weightReal) # gundam does not normalize the weight inside it.
    rans['wei'] = 1.

    result = gun.acf(gals, rans, par)
    th = result['thm']
    weightedOmega = result['wth']
    weightedOmegaErr = result['wtherr'] if doBoot else None

    return th, weightedOmega, weightedOmegaErr

def mcf_theta(omegaTh, weightedOmegaTh):
    MTheta = (1 + weightedOmegaTh)/(1 + omegaTh)
    return MTheta

def do_compute(realTab, realProperties, randTab, thMin, thNBins, thBinWidth, doRanking=True, realRaCol='RA',realDecCol='DEC',randRaCol='RA', randDecCol='Dec', doBoot=False):

    raReal = realTab[realRaCol]
    decReal = realTab[realDecCol]

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]

    th, omega, _ = omega_theta(raReal, decReal, raRand, decRand, thMin, thNBins, thBinWidth, doBoot=doBoot)

    thOmegaMcfs = np.empty((len(th), 0))

    thOmegaMcfs = np.hstack((thOmegaMcfs, th.reshape(len(th), 1)))
    thOmegaMcfs = np.hstack((thOmegaMcfs, omega.reshape(len(th), 1)))

    if len(realProperties) >= 1:

        for prop_i in realProperties:

            propNow = np.array(realTab[prop_i])

            if doRanking:
                propNowRanked = rankdata(propNow)
                weightReal = propNowRanked
            else:
                weightReal = propNow

            th, weightedOmega, _ = weighted_omega_theta(raReal, decReal, weightReal, raRand, decRand, thMin, thNBins, thBinWidth, doBoot=doBoot)

            MThetaArray = np.array(mcf_theta(omega, weightedOmega)).reshape(len(th), 1)

            thOmegaMcfs = np.hstack((thOmegaMcfs, MThetaArray))

    return thOmegaMcfs

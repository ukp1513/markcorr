import numpy as np 
from scipy.stats import rankdata
from astropy.table import Table
import os
import logging
import treecorr

logging.basicConfig(level=logging.INFO)

# gundam doesnot work for LS estimator
def omega_theta_gundam(raReal1, decReal1, raRand1, decRand1, raReal2, decReal2, raRand2, decRand2, thMin, thNBins, thBinWidth, doBoot=False):

    gals1 = Table([raReal1, decReal1], names=('ra', 'dec'))
    rans1 = Table([raRand1, decRand1], names=('ra', 'dec'))	
    
    gals2 = Table([raReal2, decReal2], names=('ra', 'dec'))
    rans2 = Table([raRand2, decRand2], names=('ra', 'dec'))

    par = gun.packpars(kind='accf', nsept=int(thNBins), septmin=thMin, dsept=thBinWidth, logsept=True, 
                       estimator='LS', doboot=doBoot) 

    gals1['wei'] = 1.
    rans1['wei'] = 1.

    gals2['wei'] = 1.
    rans2['wei'] = 1.

    result = gun.accf(gals, rans, par)
    th = result['thm']
    omega = result['wth']

    if doBoot:
        omegaErr = result['wtherr']
        return th, omega, omegaErr
    else:
        return th, omega

def omega_theta(raReal1, decReal1, raRand1, decRand1, raReal2, decReal2, raRand2, decRand2, thMin, thNBins, thBinWidth, doBoot=False):

    thMax = pow(10, (np.log10(thMin) + (thBinWidth * thNBins)))

    catReal1 = treecorr.Catalog(ra=raReal1, dec=decReal1, ra_units='deg', dec_units='deg')
    catReal2 = treecorr.Catalog(ra=raReal2, dec=decReal2, ra_units='deg', dec_units='deg')
    dd = treecorr.NNCorrelation(min_sep=thMin, max_sep=thMax, nbins=thNBins, sep_units = 'degrees')
    dd.process(catReal1, catReal2)

    catRand1 = treecorr.Catalog(ra=raRand1, dec=decRand1, ra_units='deg', dec_units='deg')
    catRand2 = treecorr.Catalog(ra=raRand2, dec=decRand2, ra_units='deg', dec_units='deg')
    rr = treecorr.NNCorrelation(min_sep=thMin, max_sep=thMax, nbins=thNBins, sep_units = 'degrees')
    rr.process(catRand1, catRand2)

    dr = treecorr.NNCorrelation(min_sep=thMin, max_sep=thMax, nbins=thNBins, sep_units = 'degrees')
    dr.process(catReal1, catRand2)

    rd = treecorr.NNCorrelation(min_sep=thMin, max_sep=thMax, nbins=thNBins, sep_units = 'degrees')
    rd.process(catReal2, catRand1)

    omega, varomega = dd.calculateXi(rr=rr, dr=dr, rd=rd)
    th = np.exp(dd.meanlogr)

    if doBoot:
        return th, omega, varomega
    else:
        return th, omega

	
# currently not being used - weighted cross-cf #TODO in future
def weighted_omega_theta(raReal, decReal, weightReal, raRand, decRand, thMin, thNBins, thBinWidth, raUnits='deg', decUnits='deg', sepUnits='degrees', doBoot=False):

    gals = Table([raReal, decReal], names=('ra', 'dec'))
    rans = Table([raRand, decRand], names=('ra', 'dec'))

    par = gun.packpars(kind='acf', nsept=int(thNBins), septmin=thMin, dsept=thBinWidth, logsept=True, 
                       estimator='LS', doboot=doBoot) 

    gals['wei'] = weightReal/np.mean(weightReal) # gundam does not normalize the weight inside it. 
    rans['wei'] = 1.

    result = gun.acf(gals, rans, par)
    th = result['thm']
    weightedOmega = result['wth']

    if doBoot:
        weightedOmegaErr = result['wtherr']
        return th, weightedOmega, weightedOmegaErr
    else:
        return th, weightedOmega
	
def mcf_theta(th, omegaTh, weightedOmegaTh):
	MTheta = (1 + weightedOmegaTh)/(1 + omegaTh)
	return MTheta
	
def do_compute(realTab1, realTab2, randTab1, randTab2, thMin, thNBins, thBinWidth, doRanking=True, 
               realRaCol1='RA',realDecCol1='DEC',randRaCol1='RA', randDecCol1='Dec', realRaCol2='RA',realDecCol2='DEC',randRaCol2='RA', randDecCol2='Dec', doBoot=False):
    
    raReal1 = realTab1[realRaCol1]
    decReal1 = realTab1[realDecCol1]

    raReal2 = realTab2[realRaCol2]
    decReal2 = realTab2[realDecCol2]

    raRand1 = randTab1[randRaCol1]
    decRand1 = randTab1[randDecCol1]

    raRand2 = randTab2[randRaCol2]
    decRand2 = randTab2[randDecCol2]

    th, omega = omega_theta(raReal1, decReal1, raRand1, decRand1, raReal2, decReal2, raRand2, decRand2, thMin, thNBins, thBinWidth)

    thOmegaMcfs = np.empty((len(th), 0))

    thOmegaMcfs = np.hstack((thOmegaMcfs, th.reshape(len(th), 1)))
    thOmegaMcfs = np.hstack((thOmegaMcfs, omega.reshape(len(th), 1)))

    return thOmegaMcfs

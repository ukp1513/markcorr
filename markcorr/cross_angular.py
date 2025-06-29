import numpy as np
import treecorr

def omega_theta(raReal1, decReal1, raRand1, decRand1, raReal2, decReal2, raRand2, decRand2, thMin, thNBins, thBinWidth, doBoot=False):
    #doBoot used as a placeholder to change to gundam in future (not relevant in case of treecorr)

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

    return th, omega, varomega

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

    th, omega, _ = omega_theta(raReal1, decReal1, raRand1, decRand1, raReal2, decReal2, raRand2, decRand2, thMin, thNBins, thBinWidth, doBoot=doBoot)

    thOmegas = np.empty((len(th), 0))

    thOmegas = np.hstack((thOmegas, th.reshape(len(th), 1)))
    thOmegas = np.hstack((thOmegas, omega.reshape(len(th), 1)))

    return thOmegas

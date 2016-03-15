#~/Code/Python/BDNYC/TDwarfplotting/modules.py

import astropy as ap, matplotlib.pyplot as plt, numpy as np
from BDNYCdb import BDdb
# db=BDdb.get_db('/Users/paigegiorla/Desktop/PG_DB_2_16_15.db')
db=BDdb.get_db('/Users/paigegiorla/Dropbox/BDNYCdb/BDNYC.db')
ma_db=BDdb.get_db('/Users/paigegiorla/Code/Models/model_atmospheres.db')

from BDNYCdb import utilities as u
import astropy.units as q
import scipy.stats as s
import warnings, glob, os, re, xlrd, cPickle, itertools, astropy.units as q, astropy.constants as ac, numpy as np, matplotlib.pyplot as plt, astropy.coordinates as apc, astrotools as a, scipy.stats as st
from random import random
from heapq import nsmallest, nlargest
from scipy.interpolate import Rbf
from pysynphot import observation
from pysynphot import spectrum
warnings.simplefilter('ignore')

def normalize_to_band(band, spectrum):
	''' normalizes spectrum to specified band by dividing by the maximum flux peak'''
	list_spectrum=list(spectrum[0])
	if band == 'Y':
		start, end = spectrum[0][0],1.143
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		band_flux = spectrum[1][a:b+1]
		spectrum[1] = spectrum[1]/max(band_flux)
		spectrum[2] = spectrum[2]/max(band_flux)
	elif band =='J':
		start, end = 1.143, 1.375
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		band_flux = spectrum[1][a:b+1]
		spectrum[1] = spectrum[1]/max(band_flux)
		spectrum[2] = spectrum[2]/max(band_flux)
	elif band == 'H':
		start, end = 1.375,2.000
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		band_flux = spectrum[1][a:b+1]
		spectrum[1] = spectrum[1]/max(band_flux)
		spectrum[2] = spectrum[2]/max(band_flux)
	elif band == 'K':
		start, end = 1.937,2.4
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		band_flux = spectrum[1][a:b+1]
		spectrum[1] = spectrum[1]/max(band_flux)
		spectrum[2] = spectrum[2]/max(band_flux)
	
	return [spectrum[0],spectrum[1],spectrum[2]]

def wavelength_band(band, spectrum):
	''' trims wavelength, flux, and unc array to indicated band for SpeX Prism spectra'''	
	list_spectrum=list(spectrum[0])
	if band == 'NIR':
		start, end = 0.9,2.3
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		spectrum=[spectrum[0][a:b+1],spectrum[1][a:b+1],spectrum[2][a:b+1]]
	if band == 'Y':
		start, end = spectrum[0][0],1.143
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		spectrum=[spectrum[0][a:b+1],spectrum[1][a:b+1],spectrum[2][a:b+1]]
	elif band =='J':
		start, end = 1.143, 1.375
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		spectrum=[spectrum[0][a:b+1],spectrum[1][a:b+1],spectrum[2][a:b+1]]
	elif band == 'H':
		start, end = 1.375,2.000
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		spectrum=[spectrum[0][a:b+1],spectrum[1][a:b+1],spectrum[2][a:b+1]]
	elif band == 'K':
		start, end = 1.937,2.4
		a = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		b = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		spectrum=[spectrum[0][a:b+1],spectrum[1][a:b+1],spectrum[2][a:b+1]]
	elif band == 'all':
		spectrum[1]=spectrum[1].value
		start, end = 0.99,1.1
		y1 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		y2 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		y = [spectrum[0][y1:y2+1],spectrum[1][y1:y2+1],spectrum[2][y1:y2+1]]

		start, end = 1.18, 1.3
		j1 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		j2 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		j = [spectrum[0][j1:j2+1],spectrum[1][j1:j2+1],spectrum[2][j1:j2+1]]

		start, end = 1.5,1.700
		h1 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		h2 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		h = [spectrum[0][h1:h2+1],spectrum[1][h1:h2+1],spectrum[2][h1:h2+1]]
		
		start, end = 2.0,2.2
		k1 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-start))
		k2 = min(range(len(list_spectrum)), key=lambda i: abs(list_spectrum[i]-end))
		k = [spectrum[0][k1:k2+1],spectrum[1][k1:k2+1],spectrum[2][k1:k2+1]]
		spectrum = []
		for i in range(len(y)):
			spectrum.append(list(y[i])+list(j[i])+list(h[i])+list(k[i]))

	return spectrum	
	
def histogram(title):
	speclist=[]
	sourcelist=[]
	sources=db.query.execute("SELECT DISTINCT spectra.source_id from spectra join spectral_types on spectra.source_id=spectral_types.source_id where spectral_types.regime='IR' and spectral_types.spectral_type>=20 and spectral_types.spectral_type<=30 and (spectra.wavelength_order=58 or spectra.wavelength_order=59 or spectra.wavelength_order=60 or spectra.wavelength_order=61 or spectra.wavelength_order=62 or spectra.wavelength_order=63 or spectra.wavelength_order=64 or spectra.wavelength_order=65)").fetchall()
	spectrals=[]
	print sources
	index=0
	while index<len(sources):
		sourcelist.append(sources[index][0])
		index=index+1
	for i in sourcelist:
		print i
		spectral=db.query.execute("select spectral_type from spectral_types where source_id='{}'".format(i)).fetchone()
		spectrals.append(spectral)
# (spectra.wavelength_order=1 or spectra.wavelength_order=2 or spectra.wavelength_order=3 or spectra.wavelength_order=4 or spectra.wavelength_order=5 or spectra.wavelength_order=6 or spectra.wavelength_order=7 or spectra.wavelength_order='n1' or spectra.wavelength_order='n2' or spectra.wavelength_order='n3' or spectra.wavelength_order='n4' or spectra.wavelength_order='n5' or spectra.wavelength_order='n6a' or spectra.wavelength_order='n6b')		
# 	(spectra.wavelength_order=58 or spectra.wavelength_order=59 or spectra.wavelength_order=60 or spectra.wavelength_order=61 or spectra.wavelength_order=62 or spectra.wavelength_order=63 or spectra.wavelength_order=64 or spectra.wavelength_order=65)
# 	instrument_id=6
	print len(spectrals)
	index=0
	while index<len(spectrals):
		speclist.append(spectrals[index][0])
		index=index+1
	print speclist
	plt.hist(speclist)
	plt.title(title)
	plt.xlabel('Spectral Type')
	plt.savefig('/Users/paigegiorla/Code/Python/BDNYC/TDwarfplotting/'+'{}'.format(title)+ '.pdf')	

def average_chisq(data):
	''' return average chisq per spectral type from data = [sptlist,chilist]
	
			Parameters:
				data : [x,y] where x and y are lists
	'''			
	from collections import defaultdict

	D = defaultdict(list)
	n = zip(*data)
	for spt,chi in n:
		D[spt].append(chi)
  
	for key,value in D.items():
		D[key] = np.average(value)
	arr = [D.keys(),D.values()]   	 
	return D,arr 	 
  	
def bin_by(x, y, half=True):
    """
    Bin x by y.
    Returns the binned "x" values and the left edges of the bins
    """
    x=np.array(x)
    y=np.array(y)
   
    if half==True:
	 y = [ '%.0f' % elem for elem in y ]
   	 bins = [20,21,22,23,24,25,26,27,28,29]
    else:
     bins = [20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29]
   
    # To avoid extra bin for the max value
    bins[-1] += 1 

    indicies = np.digitize(y, bins)

    output = []
    for i in xrange(1, len(bins)):
        output.append(x[indicies==i])

    # Just return the left edges of the bins
    bins = bins[:-1]

    return output, bins
    			
def bin_down(spectrum,jump,w1,n,spectra_id=False):
	'''
	bins down a spectrum to a resolution defined by a jump(bin length), w1 (first wavelength point of lowest resolution image), and i(number of total points desired)
	'''
# 	if spectra_id==False:
 	W,F,U=spectrum[0],spectrum[1],spectrum[2]
# 	else:
# 		name,W,F,U=db.query.execute("SELECT source_id,wavelength,flux,unc from spectra where id={}".format(spectrum)).fetchone()
	binned_wl=0																	#defined by the wavelength points of HD19467B data
	start=w1-(jump/2) 															#start half way before 1st flux point
	F=F/max(F)	
	U=U/max(F)
	W,F,U=u.scrub([W,F,U])

	original=np.array([W,F,U])
	flux2=0
	W_temp, F_temp, U_temp=[],[],[]
	unc2=0																	#first wl point is at start+the step/2
	wavelength=0

#!		loop through 21 iterations for the amount of wavelength ranges I have
	for i in range(n):
		binned_wl=jump + start 															#range is half below and half above w1
		
		flux2=sum(F[np.where(np.logical_and(W>start,W<binned_wl))])					#binning flux
		unc2=np.sqrt(sum(U[np.where(np.logical_and(W>start,W<binned_wl))]**2))			#bin uncertainty in quadrature
		
		start=binned_wl																	#set new range start
	
		F_temp.append(flux2)
		U_temp.append(unc2)
		wavelength=w1+(i*jump)															#this is where the flux point will go on plot
		W_temp.append(wavelength)	

	template=np.array([W_temp, F_temp, U_temp])
	return original, template
	
def bin_down_model(W,F,jump,w1,n):
	'''
	bins down a model spectrum to a resolution defined by a jump(bin length), w1 (first wavelength point of lowest resolution image), and n(number of total points desired)
	'''
	binned_wl=0																	#defined by the wavelength points of HD19467B data
	start=w1-(jump/2) 		#start half way before 1st flux point
	maxFm=max(F)
	F=F/maxFm										
	W,F=scrub([W,F])
	W=W.value
	F=F.value
	original=np.array([W,F])
	flux2=0
	W_temp, F_temp=[],[]
	unc2=0																	#first wl point is at start+the step/2
	wavelength=0

#!		loop through 21 iterations for the amount of wavelength ranges I have
	for i in range(n):
		binned_wl=jump + start 															#range is half below and half above w1
		
		flux2=sum(F[np.where(np.logical_and(W>start,W<binned_wl))])					#binning flux
		
		start=binned_wl																	#set new range start
	
		F_temp.append(flux2)
		wavelength=w1+(i*jump)															#this is where the flux point will go on plot
		W_temp.append(wavelength)	

	template=np.array([W_temp, F_temp])
	return original, template
	
def goodness(spec1, spec2, array=False, exclude=[], filt_dict=None, weighting=True, verbose=False):
  if isinstance(spec1,dict) and isinstance(spec2,dict) and filt_dict:
    bands, w1, f1, e1, f2, e2, weight, bnds = [i for i in filt_dict.keys() if all([i in spec1.keys(),i in spec2.keys()]) and i not in exclude], [], [], [], [], [], [], []
    for eff,b in sorted([(filt_dict[i]['eff'],i) for i in bands]):
      if spec1[b] and spec1[b+'_unc'] and spec2[b]: bnds.append(b), w1.append(eff), f1.append(spec1[b]), e1.append(spec1[b+'_unc']), f2.append(spec2[b]), e2.append(spec2[b+'_unc'] if b+'_unc' in spec2.keys() else 0*spec2[b].unit), weight.append((filt_dict[b]['max']-filt_dict[b]['min']) if weighting else 1)
    bands, w1, f1, e1, f2, e2, weight = map(np.array, [bnds, w1, f1, e1, f2, e2, weight])
    if verbose: printer(['Band','W_spec1','F_spec1','E_spec1','F_spec2','E_spec2','Weight','g-factor'],zip(*[bnds, w1, f1, e1, f2, e2, weight, weight*(f1-f2*(sum(weight*f1*f2/(e1**2 + e2**2))/sum(weight*f2**2/(e1**2 + e2**2))))**2/(e1**2 + e2**2)]))
  else:
    spec1, spec2 = [[i.value if hasattr(i,'unit') else i for i in j] for j in [spec1,spec2]]
    if exclude: spec1 = [i[idx_exclude(spec1[0],exclude)] for i in spec1]
    (w1, f1, e1), (f2, e2), weight = spec1, rebin_spec(spec2, spec1[0])[1:], np.gradient(spec1[0])
    if exclude: weight[weight>np.std(weight)] = 0
  C = sum(weight*f1*f2/(e1**2 + e2**2))/sum(weight*f2**2/(e1**2 + e2**2))
  G = weight*(f1-f2*C)**2/(e1**2 + e2**2)
  if verbose: plt.loglog(spec2[0], spec2[1]*C, 'r', label='spec2', alpha=0.6), plt.loglog(w1, f1, 'k', label='spec1', alpha=0.6), plt.loglog(w1, f2*C, 'b', label='spec2 binned', alpha=0.6), plt.grid(True), plt.legend(loc=0)
  return [G if array else sum(G), C]

def rebin_spec(spec, wavnew):
  Flx,  filt = spectrum.ArraySourceSpectrum(wave=spec[0], flux=spec[1]), spectrum.ArraySpectralElement(spec[0], np.ones(len(spec[0])))
  return observation.Observation(Flx, filt, binset=wavnew, force='taper').binflux

def montecarlo(object, idlist, N=100, save=''):
	G = []
	spts=[]
	sptlist=[]
	for p in idlist:
		spt=db.query.execute("SELECT spectral_types.spectral_type from spectral_types join spectra on spectral_types.source_id=spectra.source_id where spectral_types.regime='IR' and spectra.id='{}'".format(p)).fetchone()
		spts.append(spt)
	index=0
	while index<len(spts):
		sptlist.append(spts[index][0])
		index=index+1		

	
	for j in range(len(idlist)):
		original, model=bin_down(idlist[j],0.0267097,1.1826776,21)
# 		g=[]
# 		r=np.random.normal(model[1], model[2],size=len(model[1]))
# 		for _ in itertools.repeat(None,N): g.append((goodness(object, [model[0], r , model[2]])[0], sptlist[j]))
# 		G.append(g)
		
		g=[]
		r=np.random.normal(object[1], object[2],size=len(object[1]))
		for _ in itertools.repeat(None,N): g.append((goodness(model, [object[0], r , object[2]])[0],sptlist[j]))
		G.append(g)	
		
#  		plt.errorbar(model[0],r,yerr=model[2])
#  	moodel=[model[0],r,model[2]]
#  	H=zip(*G)
# 	fits=np.array(H[0])
#  	bests=fits.argsort()[:10]
#  	print bests
	return G
	

def load_bt_settl_spectra(year=2013):
  syn, files, bt_settl = BDdb.get_db(path+'Models/model_atmospheres.db'), glob.glob(path+'Models/BT-Settl_M-0.0_a+0.0_{}/*.spec.7'.format(year)), [] 
   
  def read_btsettl(filepath):
    obj, Widx, (T, G) = {}, [], map(float, os.path.splitext(os.path.basename(filepath))[0].replace('lte','').split('-')[:2])
    T, data = int(T*100), open(filepath, 'r')
    lines = [i.split()[:3] for i in data]
    data.close()
    W, F = [[i[idx] for i in lines[::20]] for idx in [0,1]]
    W = (np.array([float(w) for w in W])*q.AA).to(q.um)
    for n,w in enumerate(W):
      if (w.value <= 0.3) or (w.value >= 30.0): Widx.append(n)                                                     
    W, F = np.delete(W,Widx)*q.um, np.delete(F,Widx)
    return [T, G, W, ((10**np.array([float(f.replace('D','E')) for f in F]))*q.erg/q.s/q.cm**3).to(q.erg/q.s/q.cm**2/q.AA)]
    
  for f in files:
    try:
      obj = read_btsettl(f)
      syn.query.execute("INSERT INTO bt_settl_{} VALUES (?, ?, ?, ?, ?)".format(year), (None, obj[0], obj[1], obj[2].value, obj[3].value)), syn.modify.commit()
      print "{} {}".format(obj[0], obj[1])
    except: print "Failed: {} {}".format(obj[0], obj[1])
  syn.modify.close() 
 
def polynomialfit(data, constraint_on_x,order):
	''' return polynomial that is fit to PRE-SORTED data
	
			Parameters:
				data : [x,y] where x and y are lists
				constraint_on_x: is [a,b] where a and b are the limiting factors to fit 
	'''
	b=[]
	n=zip(*data)
	for i in range(len(n)):
		if n[i][0]>=constraint_on_x[0] and n[i][0]<=constraint_on_x[1]:
			b.append(n[i])
	b=zip(*b)	
	pfit = np.polyfit(b[0],b[1], order)   # Fit a 3rd order polynomial to (x, y) data
	yfit = np.polyval(pfit, b[0])   # Evaluate the polynomial at x
	return yfit,b
 
def redchisq(ydata,ymod,deg=2,unc=None):    
	chisq=0
	if unc==None:  
		chisq=np.sum((ydata-ymod)**2)  
	else:  
		diff=ydata-ymod
		div=diff/unc
		square=(div)**2
		chisq=np.sum(square) 
	# Number of degrees of freedom assuming 2 free parameters  
# 	nu=ydata.size-1-deg  
	return chisq
	
  
def scrub(data):
  '''
  For input data [w,f,e] or [w,f] returns the list with NaN, negative, and zero flux (and corresponsing wavelengths and errors) removed. 
  '''
  data = [i*q.Unit('') for i in data]
  data = [i[np.where(np.logical_and(data[1].value>0,~np.isnan(data[1].value)))] for i in data]
  data = [i[np.unique(data[0], return_index=True)[1]] for i in data]
  return [i[np.lexsort([data[0]])] for i in data]  
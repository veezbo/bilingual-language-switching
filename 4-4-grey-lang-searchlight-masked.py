import mvpa2
import mvpa2.suite as M # machine learning with brain data
import numpy as N
import pylab as P # plotting (matlab-like)
import os # file path processing, directory contents
import datetime # time stamps
import pickle # cPickle
import gzip
import numpy as np
from mvpa2 import cfg
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.measures.base import CrossValidation
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.testing.datasets import datasets
from mvpa2.mappers.fx import mean_sample
from mvpa2.datasets.mri import map2nifti
from mvpa2.support.pylab import pl
from mvpa2.misc.plot.lightbox import plot_lightbox
#from mvpa.misc.data_generators import noisy_2d_fx
#from mvpa.mappers.svd import SVDMapper
#from mvpa.mappers.mdp_adaptor import ICAMapper, PCAMapper
#import mvpa.cfg as cfg

#OUTLINE
# check data not already pickled - if yes reload; if no load from scratch and pickle
# try to clear/delete/garbage collect the original much larger data set
# convert stimulus conditions to class conditions
# print five sample timecourses + sample similarity by chunks and by condition
# do detrending, and repeat
# do zscores and repeat

# later - do boxcar to average over events

if __debug__:
    M.debug.active += ["SLC"]
numFolds=6

# sessionPath = '/home/brain/host/20120730jinyq-lang/'
sessionPath = os.getcwd()
#'/mnt/data/brianDataStore/MRI/vanillaMarginsChinese/19871021HGLI_201010290800'
#'/mnt/data/brianDataStore/MRI/vanillaMarginsItalian/19790514GOPT_201011161600'
preprocessedCache = os.path.join(sessionPath, 'detrendedZscoredMaskedPyMVPAdataset.pkl')
trimmedCache = os.path.join(sessionPath, 'averagedDetrendedZscoredMaskedPyMVPAdataset.pkl')

# LOAD DATASET (Following the instruction of Brian, "and False" is added at lines 29 and 33.)
if os.path.isfile(trimmedCache) and False: 
	print 'loading cached averaged, trimmed, preprocessed dataset',trimmedCache,datetime.datetime.now()
	dataset = pickle.load(gzip.open(trimmedCache, 'rb'))
else:
	if os.path.isfile(preprocessedCache) and False: 
		print 'loading cached preprocessed dataset',preprocessedCache,datetime.datetime.now()
		dataset = pickle.load(gzip.open(preprocessedCache, 'rb', 5))
	else:
		# if not, generate directly, and then cache
		print 'loading and creating dataset',datetime.datetime.now()
		# chunksTargets_boldDelay="chunksTargets_boldDelay4-4.txt" #Modified
		chunksTargets_boldDelay="chunksTargets_boldDelay3-4-Japanese_English.txt" #Modified
		boldDelay=3 #added
		stimulusWidth=4 #added
		volAttribrutes = M.SampleAttributes(os.path.join(sessionPath,'behavioural',chunksTargets_boldDelay)) # default is 3.txt.
		dataset = M.fmri_dataset(samples=os.path.join(sessionPath,'analyze/functional/functional4D.nii'),
			targets=volAttribrutes.targets, # I think this was "labels" in versions 0.4.*
			chunks=volAttribrutes.chunks,
			mask=os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr'),
                        add_fa={'vt_thr_glm': os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr')})# added for searchlight

		# DATASET ATTRIBUTES (see AttrDataset)
		print 'functional input has',dataset.a.voxel_dim,'voxels of dimesions',dataset.a.voxel_eldim,'mm'
		print '... or',N.product(dataset.a.voxel_dim),'voxels per volume'
		print 'masked data has',dataset.shape[1],'voxels in each of',dataset.shape[0],'volumes'
		print '... which means that',round(100-100*dataset.shape[1]/N.product(dataset.a.voxel_dim)),'% of the voxels were masked out'
		print 'of',dataset.shape[1],'remaining features ...'
		print 'summary of conditions/volumes\n',datetime.datetime.now()
		print dataset.summary_targets()
		# could add use of removeInvariantFeatures(), but takes a long time, and makes little difference if mask is working well

		## TEMP TEMP TEMP
		## just to make things faster for illustration purposes, we save some memory by 
		#print 'TEMP: trimming dataset to reduce memory footprint - for illustration only'
		#dataset = dataset[dataset.chunks < 5] # first 4 runs
		#dataset = dataset[:,range(0,dataset.shape[1],3)] # 33% sample of features (1 in 3)
		#dataset.samples = dataset.samples.astype('float')

		# PLOT RAW DATA
		#P.figure()
		#P.subplot(221)
		#randomVoxs = range(1,dataset.shape[1],dataset.shape[1]/5)
		#P.plot(dataset.samples[:,randomVoxs[0]]); P.plot(dataset.samples[:,randomVoxs[1]]); P.plot(dataset.samples[:,randomVoxs[2]]); P.plot(dataset.samples[:,randomVoxs[3]]); P.plot(dataset.samples[:,randomVoxs[4]]); 
		#P.title('five random voxel timecourses')

		# DETREND
		print 'detrending (remove slow drifts in signal, and jumps between runs) ...',datetime.datetime.now() # can be very memory intensive!
		M.poly_detrend(dataset, polyord=1, chunks_attr='chunks') # linear detrend
		print '... done',datetime.datetime.now()

		#P.subplot(222)
		#P.plot(dataset.samples[:,randomVoxs[0]]); P.plot(dataset.samples[:,randomVoxs[1]]); P.plot(dataset.samples[:,randomVoxs[2]]); P.plot(dataset.samples[:,randomVoxs[3]]); P.plot(dataset.samples[:,randomVoxs[4]]); 
		#P.title('... same, detrended')

		# ZSCORE
		print 'zscore normalising (give all voxels similar variance) ...',datetime.datetime.now()
		M.zscore(dataset, chunks_attr='chunks', param_est=('targets', ['base'])) # zscoring, on basis of rest periods
		print '... done',datetime.datetime.now()
		#P.savefig(os.path.join(sessionPath,'pyMVPAimportDetrendZscore.png'))
		print 'saving as compressed file',preprocessedCache
		#dataset.save(os.path.join(sessionPath, 'detrendedZscoredMaskedPyMVPAdataset.pkl'), compression=5) # saw no difference in time or space with compression=9
		#with gzip.open(preprocessedCache, 'wb', 5) as pickleFile:
		#	pickle.dump(dataset, pickleFile)
		#pickleFile = gzip.open(preprocessedCache, 'wb', 5);
		#pickle.dump(dataset, pickleFile);

	#P.subplot(223)
	#P.plot(dataset.samples[:,randomVoxs[0]]); P.plot(dataset.samples[:,randomVoxs[1]]); P.plot(dataset.samples[:,randomVoxs[2]]); P.plot(dataset.samples[:,randomVoxs[3]]); P.plot(dataset.samples[:,randomVoxs[4]]); 
	#P.title('... same, zscore normalised')

	# AVERAGE OVER MULTIPLE VOLUMES IN A SINGLE TRIAL
	print 'averaging over trials ...',datetime.datetime.now()
	dataset = dataset.get_mapped(M.mean_group_sample(['chunks','targets']))
	print '... only',dataset.shape[0],'cases left now'
	dataset.chunks = N.mod(N.arange(0,dataset.shape[0]),5)

	##P.subplot(224)
	##P.plot(dataset.samples[:,randomVoxs[0]]); P.plot(dataset.samples[:,randomVoxs[1]]); P.plot(dataset.samples[:,randomVoxs[2]]); P.plot(dataset.samples[:,randomVoxs[3]]); P.plot(dataset.samples[:,randomVoxs[4]]); 
	##P.title('... same, trial averaged')
	#print 'saving as compressed file'
	#dataset.save(os.path.join(sessionPath, 'trialaveragedDetrendedZscoredMaskedPyMVPAdataset.gzipped.hdf5'), compression=5) # saw no difference in time or space with compression=9

	# REDUCE TO CLASS LABELS, AND ONLY KEEP CONDITIONS OF INTEREST (MAMMALS VS TOOLS)
	dataset.targets = [t[0] for t in dataset.targets]
	dataset = dataset[N.array([l in ['c', 'k'] for l in dataset.sa.targets], dtype='bool')]
	print '... and only',dataset.shape[0],'cases of interest (Chinese vs Korean)'
	dataset=M.datasets.miscfx.remove_invariant_features(dataset)
	print 'saving as compressed file',trimmedCache
	pickleFile = gzip.open(trimmedCache, 'wb', 5);
	pickle.dump(dataset, pickleFile);

# DO LEARNING AND CLASSIFICATION
# define classifier
# NOTE!!! in 0.6.0 CrossValidatedTransferError and TransferError are combined into a new object called M.CrossValidation (?)
# Sometimes Akama added cv=3 temporally.
anovaSelectedSMLR = M.FeatureSelectionClassifier(
	M.PLR(),
	M.SensitivityBasedFeatureSelection(
		M.OneWayAnova(),
		M.FixedNElementTailSelector(500, mode='select', tail='upper')
	),
)
foldwiseCvedAnovaSelectedSMLR = M.CrossValidation(
	anovaSelectedSMLR,
	M.NFoldPartitioner(),
	enable_ca=['samples_error','stats', 'calling_time','confusion']
)

center_ids = dataset.fa.vt_thr_glm.nonzero()[0]

plot_args = {
    'background' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/anat.nii.gz'),
    'background_mask' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/mask_brain.nii.gz'),
    'overlay_mask' : os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr'),
    'do_stretch_colors' : False,
    'cmap_bg' : 'gray',
    'cmap_overlay' : 'autumn', # pl.cm.autumn
    'interactive' : cfg.getboolean('examples', 'interactive', True),
    }

for radius in [3]:
# tell which one we are doing
    print 'Running searchlight with radius: %i ...' % (radius)

    sl = sphere_searchlight(foldwiseCvedAnovaSelectedSMLR, radius=radius, space='voxel_indices',
                            center_ids=center_ids,
                            postproc=mean_sample())

    ds = dataset.copy(deep=False,
                      sa=['targets', 'chunks'],
                      fa=['voxel_indices'],
                      a=['mapper'])

    sl_map = sl(ds)
    sl_map.samples *= -1
    sl_map.samples += 1

    niftiresults = map2nifti(sl_map, imghdr=dataset.a.imghdr)
    niftiresults.to_filename(os.path.join(sessionPath,'analyze/functional/Plang-grey-searchlight.nii'))
    print 'Best performing sphere error:', np.min(sl_map.samples)
#print dataset.niftihdr['scl_slope']
#print dataset.map2Nifti(dataset.samples).header['scl_slope']

    #fig = pl.figure(figsize=(25, 15), facecolor='white')#figsize used to be (12,4)
   # subfig = plot_lightbox(overlay=niftiresults,
                           #vlim=(0.5, None), slices=None,
                           #fig=fig, **plot_args)
    #pl.title('Accuracy distribution for radius %i' % radius)

# run classifier
#print 'learning on detrended, normalised, averaged, mammals vs tools ...',datetime.datetime.now()
#results = foldwiseCvedAnovaSelectedSMLR(dataset)
#print '... done',datetime.datetime.now()
#print 'accuracy',N.round(100-N.mean(results)*100,1),'%',datetime.datetime.now()

#New lines for out putting the result into a csv file.
#precision=N.round(100-N.mean(results)*100,1)
#st=str(boldDelay) + ',' + str(stimulusWidth) + ',' + str(precision) +'\n'
#f = open( "withinPredictionResult.csv", "a" )
#f.write(st)
#f.close


# display results
#P.figure()
#P.title(str(N.round(foldwiseCvedAnovaSelectedSMLR.ca.stats.stats['ACC%'], 1))+'%, n-fold SMLR with anova FS x 500')
#foldwiseCvedAnovaSelectedSMLR.ca.stats.plot()
#P.savefig(os.path.join(sessionPath,'confMatrixAvTrialMammTools.png'))
#print foldwiseCvedAnovaSelectedSMLR.ca.stats.matrix

#print 'accuracy',N.round(foldwiseCvedAnovaSelectedSMLR.ca.stats.stats['ACC%'], 1),'%',datetime.datetime.now()

# test
#weightAnalyser = anovaSelectedSMLR.get_sensitivity_analyzer(postproc=M.maxofabs_sample())
#learnerWeightsAmalgFolds = weightAnalyser(dataset) # gives only positive values, cos takes maxofabs...
#sum(learnerWeightsAmalgFolds.samples > 0) # number of non-zero values
#learnerWeightedVoxels = dataset.fa.voxel_indices[(learnerWeightsAmalgFolds.samples > 0)[0]]
#print 'learnerWeightedVoxels', learnerWeightedVoxels
#pickle.dump(learnerWeightedVoxels,open(os.path.join(sessionPath,'behavioural','20110224-savetxt-60-learnerWeightedVoxels.txt'),'w'))

#new code tentative
#N.savetxt(os.path.join(sessionPath,'behavioural','20110414yn-savetxt-learnerWeightedVoxels.txt'),learnerWeightedVoxels, delimiter=',')
#N.savetxt(os.path.join(sessionPath,'behavioural','20110422kw-savetxt-60-learnerWeightedVoxels.txt'),learnerWeightedVoxels, delimiter=',', fmt='%d')
# this should give average anova measure over the folds - but in fact would be much the same as taking over single fold
#sensana = anovaSelectedSMLR.get_sensitivity_analyzer(postproc=M.maxofabs_sample())
#cv_sensana = M.RepeatedMeasure(sensana, M.NFoldPartitioner())
#sens = cv_sensana(dataset)
#print sens.shape
#M.map2nifti(dataset, N.mean(sens,0)).to_filename("anovaSensitivity_"+sessionID+'.nii')


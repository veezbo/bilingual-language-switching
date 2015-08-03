import os # paths, directory contents
import re # string matching
import codecs # reading encoded files (see notes at end)
import sys # early exit, for dev and debugging only
import copy # allows deep copy of numpy array
import numpy as N
import scipy.io as Sio

# PURPOSE AND USAGE
#
# SPM allows you to supply the conditions for each scanning sequence as a matlab *.mat file, as described here:
#		This  *.mat  file  must  include  the following cell arrays (each 1 x n): names, onsets and durations. eg. names=cell(1,5), onsets=cell(1,5), durations=cell(1,5), then names{2}='SSent-DSpeak', onsets{2}=[3 5 19 222], durations{2}=[0  0  0 0], contain the required details of the second condition. These cell arrays may be made available by your stimulus delivery program, eg. COGENT. The duration vectors can contain a single entry if the durations are identical for all events.
#
# This program will read through a single text log file from E-Prime, and produce one conditionsX_run_.mat file for each scanning run/sequence
#
# To use this script, set the correct base directory for the scanning session (which should have the eprime text files in the behaviour subdirectory, and an spmStats directory to hold the condition parameter output), the condition names used in the eprime file to signal each event type of interest, the number of trials per run, and the other volume counts (length of stimulus, trial duration, and starting and finishing rest periods)

# PARAMETERS

# sessionPath = 'D:\\2014GLM\\20140604-ws' # base directory where I keep all data from experimental session (MRI images, behavioural data, scripts, statistical analysis results
sessionPath = os.getcwd()
statsPath = os.path.join(sessionPath,'spmStats')
behavFilename = [f for f in os.listdir(os.path.join(sessionPath,'behavioural')) if f.endswith('-log.txt')][0] # guess that the first text file in the behavioural directory is the ePrime output

categoryMarkers = { 'English': 'E',	# single character markers for three stimulus category conditions
                        'Japanese': 'J'}
conditionNames = categoryMarkers.keys() # 

trialsPerRun = 40 # number of trials in each run

trialVols = 10.0 # number of volumes per stimulus trials
stimulusVols = 3.0 # number of volumes containing stimulus-specific activity (remaining trial volumes assumed to be rest)
runInVols = 0 # number of baseline volumes at start of scanning run (= a PyMVPA chunk)
# run in was 15 vols/secs for main experiment, but at very start was 16 (15.5 baseline, plus 0.5 sec prefixation), I think
runOutVols = 0 # number of baseline volumes at end of scanning run

# PREPARE MATLAB-COMPATIBLE STRUCTURE FOR CONDITIONS 

conditionNames = categoryMarkers.keys() # 
numConditions = len(conditionNames)

conditionParams = {} # dictionary to store condition parameters
conditionIndex = {} # ... and its condition name to index mapping
conditionParams['names'] = N.zeros((numConditions,), dtype=N.object) # was (3,), worked but in column. if (1,3) then serial, but not sure if column actually required
conditionParams['durations'] = N.zeros((numConditions,), dtype=N.object) # ditto
conditionParams['onsets'] = N.zeros((numConditions,), dtype=N.object) # ditto

for c in range(numConditions):
	conditionIndex[conditionNames[c]] = c # reverse condition names, so can find right condition number on basis of label found in behavioural file
	conditionParams['names'][c] = conditionNames[c] # use [0,c] if dimensionality if (1,3), or [c] if (3,)
	conditionParams['durations'][c] = [stimulusVols] # NOTE: if you put in bare numbers (ie not enclosed in array), you get an empty result in matlab
	conditionParams['onsets'][c] = [] # will append real values as we go along [10.0*c, 20.0*c, 30.0*c]
emptyOnsetsBuffer = copy.deepcopy(conditionParams['onsets']) # empty copy so can reset at start of each new chunk ... if use equal, just get reference copy...

# GO THROUGH FILE FINDING START OF EVENTS IN EACH CONDITION

trialCounter = 0; # keeps track of number of stimuli, for knowing where to put in run-breaks
chunkCounter = 0; # chunk (run) counter - used by PyMVPA for detrending, and for cross-validation partitions
concept = ''
category =''
#print 'opening behavioural log file',behavFilename
behavFile = codecs.open(os.path.join(sessionPath,'behavioural',behavFilename), 'r', 'utf_16') # actually 'utf_16' is enough, don't need 'utf_16_le'
for line in behavFile:
	if re.search('Language: ',line): #not 'Category: ` Hiroyuki Akama added on 2012/9/18
		category=re.search('Language: (.+)\r',line).groups(0)[0]#not 'Category: ` Hiroyuki Akama added on 2012/9/18
		if trialCounter % trialsPerRun == 0:
			if chunkCounter > 0:
				# AT END OF EACH CHUNK, SAVE STRUCTURE FOR IMPORT INTO MATLAB
				#print "... CHUNK END>", chunkCounter
				conditionsFile = os.path.join(statsPath,'conditionsX'+str(numConditions)+'run'+str(chunkCounter)+'.mat')
				print "saving data from run", chunkCounter, "to", conditionsFile
				Sio.savemat(conditionsFile, conditionParams)
				conditionParams['onsets'] = copy.deepcopy(emptyOnsetsBuffer)
			chunkCounter += 1;
			#print "<CHUNK START", chunkCounter
		onsetVol = runInVols+trialVols*(trialCounter%trialsPerRun)
		conditionParams['onsets'][conditionIndex[category]].append(onsetVol)
		print "\t[%i] [%i] %s %i %i s" % (trialCounter, trialCounter%trialsPerRun, category, chunkCounter, onsetVol) 
		trialCounter += 1
#print "CHUNK END>", chunkCounter
conditionsFile = os.path.join(statsPath,'conditionsX'+str(numConditions)+'run'+str(chunkCounter)+'.mat')
print "saving data from run", chunkCounter, "to", conditionsFile
Sio.savemat(conditionsFile, conditionParams)







# go through file
# at each event, increment onset time
# if of condiction X, append on array of that 

# keep onsets for each chunk in dictionary
# at new chunk, write out current data, and then empty onsets
# at each trial, append to onsets



"""
A cheap version of univariate Searchlight of which the output is to be compared with the GLM contrast maps.
The z-score of the classification accuracy values for each voxel in a mean searchlight image is to be computed.
Setting the threshold of p-value is set to 0.001 under the hypothesis of normal distribution,
only the z-scores >3.08 are extracted and kept in the original Searchlight map.
Needs nibabel.

param value to be changed.
img = nibabel.load('/host/brain/searchlight/mammal_tool_searchlight.nii')
Searchlight nii file to be input.

filename='/host/brain/searchlight/univariate_mammal_tool_searchlight.nii'
as an out put

"""
import numpy as np
import pylab # as P # plotting (matlab-like)
import csv
import sys
import scipy
import nibabel
import os

classificationName = "Japanese_English"
# img = nibabel.load('/mnt/host/Hatada-ex/20140909-ys-op/analyze/functional/Pmt-grey-searchlight-r1.nii')
img = nibabel.load(os.path.join(os.getcwd(), "analyze/functional/%s-grey-searchlight.nii" % classificationName))
#load an nii file. In this case, the model-constructor searchlight result with the raidius =1 covering the grey matter mask.
imgdata=img.get_data()
flattened_imgdata=np.ndarray.flatten(imgdata)
non_zero_data=[x for x in flattened_imgdata if not x==0] # Extract the voxels of which the accuracy is not 0.
print len(non_zero_data)
mean_non_zero_data=np.mean(non_zero_data)
print 'The mean is', mean_non_zero_data
sd_non_zero_data=np.std(non_zero_data)
print 'The standard deviation is', sd_non_zero_data

#1/0
#technique to stop python

#zscore_non_zero_data=np.zeros((53,63,23))
#print zscore_non_zero_data

imgdatanew=imgdata

for i in range(np.shape(imgdata)[0]):
    for j in range(np.shape(imgdata)[1]):
        for k in range(np.shape(imgdata)[2]):
            if (imgdata[i,j,k]==0):
                imgdatanew[i,j,k]=0;
            else:
                tmp=(float)(imgdata[i,j,k]-mean_non_zero_data)/sd_non_zero_data
                if tmp>3.08: #P(z>3.08)=0.001 on the standard normal curve. Used a z-score cummulative table.
                    imgdatanew[i,j,k]=tmp
                else:
                    imgdatanew[i,j,k]=0

print imgdatanew
#1/0

print np.any(x>0 for x in imgdatanew)
#Return true.

filename=os.path.join(os.getcwd(), 'analyze/functional/%s-univariate-Pmt-grey-searchlight-r1.nii' % classificationName)
newimg = nibabel.Nifti1Image(imgdatanew, None, img.get_header()) 
newimg.to_filename(filename)
print newimg

#The file created here is 'univariate_mean_searchlightgreyr1_01.nii'.



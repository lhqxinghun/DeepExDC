#!/bin/bash

config=$1
echo $config
cd src/DeepExDC




echo "
#                  STEP 1                  #
# Compartment score calculation
# select the best pc out of PC1 and PC2 for each chromosome 
# by comparing each one against GC content and gene density through correlation
# input:  scHi-C matrix files(.npy),annotation files(cytoBand.txt,*.chrom.sizes,*.fa,*.refGene.gtf.gz)
#         chr start end count
# output: scHi-C compartment score
#         presudo-bulk compartment score(.hdf5)
echo python compartment_score.py -c $config
# STEP 1 ENDS
"
# python compartment_score.py -c $config

echo "
#                  STEP 2                  #
# classification of single cells
# A one-dimensional convolutional neural network is designed for the classification of single cells
# It accepts the calculated compartment scores of single cells
# and learns to predict the types of conditions that they are under
# so that the network acquires an ability to distinguish these cells from different conditions in the light of compartment scores.
input:  scCompartment score file(.hdf5),label file
output  the trained classification network model(.h5)
echo python train.py -c $config
# STEP 2 ENDS
"
python train.py -c $config

echo "
#                  STEP 3                 #
# shapley explanation
# A deep additive explainer is constructed based on the trained classification network with the help of SHAP
# a vector of SHAP values is obtained for every compartment bin of each cell
# and the elements in this vector reflect the contributions of this compartment bin to all the possible condition types of this cell
echo python deepexplation.py -c $config
# STEP 3 ENDS
"
python deepexplation.py -c $config

echo "
#                  STEP 4                 #
# determination of differential analysis
# A distance matrix between pairwise cells is created for 
# each compartment bin in the space spanned by SHAP value vector
# and the q-values and effect sizes given by distance-based PERMANOVA 
# can be used to determine the differential compartment bins with significant changes across conditions. 
echo python getdiffcompartments.py -c $config
# STEP 4 ENDS
"
python getdiffcompartments.py -c $config

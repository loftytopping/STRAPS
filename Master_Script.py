##########################################################################################
#											 #
#    Fitting machine learning algorithms to predict a given mass spectra based on        #
#    distinct m/z channels. This relies on the Scikit learn libraries as defined         #
#    by the import statements given below. More information is given at the beginning    #
#    of each section.                                                                    # 
#                                                                                        #
#                                                                                        #
#    Copyright (C) 2016  David Topping : david.topping@manchester.ac.uk                  #
#                                      : davetopp80@gmail.com     			 #
#    Personal website: davetoppingsci.com                                                #
#											 #
#    Evaluated jointly with:								 #
#    James Allan : James.Allan@manchester.ac.uk						 #
#    Rami Alfarra : rami.alfarra@manchester.ac.uk					 #
#                                                                                        # 
#    This program is free software: you can redistribute it and/or modify                #
#    it under the terms of the GNU Affero General Public License as published            #
#    by the Free Software Foundation, either version 3 of the License, or                #
#    (at your option) any later version.                                                 # 
#                                                                                        #   
#    This program is distributed in the hope that it will be useful,                     #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                      # 
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       #
#    GNU Affero General Public License for more details.                                 #
#                                                                                        #
#    You should have received a copy of the GNU Affero General Public License            # 
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.               #
#                                                                                        #
#                                                                                        #
#                                                                                        # 
#                                                                                        #
##########################################################################################

##########################################################################################
# Introduction
# The idea behind this script is as follows

# We have a database of EI mass spectra, which gives peak height for any given
# m/z channel, per compound. Each compound is represented by a SMILES string which
# is a way of representing a chemical structure For more information on SMILES see:
# http://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
# https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system

# The question we want to answer is: Can we predict EI mass spectra based on 
# the structure of a molecule?? To answer this we take the database of measured
# spectra and then extract structural features per molecules to see if a range of
# machine learning algorithms can use these features to fit a model that relates
# peak height to such features. How we extract these features can vary. We normally use
# a database of SMARTS strings to interrogate SMILEs strings. These SMARTS strings
# represent key features. For example, this might include CH2 groups, a COOH group
# next to a CH2 group, and so on. There are many databases that collate such strings
# and we use a few in this study. We use our own SMARTS libraries based on
# models of chemical properties such as vapour pressure as well as more generic
# libraries defined elsewhere, referred to as MACCS keys or FP4 keys:
#
# http://www.dalkescientific.com/writings/NBN/fingerprints.html
#
# To begin training the models, we want a matrix of identified features per compound
# as follows:
# SMILES ---[use SMARTS]---> a matrix of identified features ---> input to machine
# learning algorithms to relate peak height to features ---> a model that can predict
# peak height when given a SMILES string.

# As you look through this code you will see we will use the UManSysProp facilities
# to use some of these SMARTS libraries, whilst for more generic SMARTS we extract SMARTS from
# the RDKit python package. This is simply for efficiency.
#
# The code presented here focuses on the aerosol mass spectrometer. Please see the paper
# associated with this development for further details. The original data was taken from:
# http://cires1.colorado.edu/jimenez-group/AMSsd/
# Where each file now has a SMILES string inserted.
##########################################################################################

##########################################################################################
# Procedure used in the following code
# 0. Specific options used throughout the code.
# 1. Read in EI spectra from .itx files [common with AMS]
# 2. Remove channels with less than 1% total peak height or negative values
# 3. Generate MACCS [or other] fingerprint per compound and plot coverage (optional). This
#    gives us an idea of the sparsity of information extracted using a given fingerprint
# 4. Variable selection (optional) - extract only those features deemed
#    important to predict peak heights. This is based on an ensemble classifier
#    and refines the generated feature matrix / list.
# 5. Train a model to predict the occurance of m/z channels (non-optional). If this
#    isnt selected, every channel is used to train an algorithm to predict peak
#    heights. In most cases, the occurance of data in any given m/z channel is
#    compound specific [expected]. However a trained algorithm might still
#    predict information in a given channel even if none is there - it depends
#    on how much information it has gained friom the training set. By using a 
#    model to predict the presence of any non-zero information, before training
#    one to peak height, I have found it can drastically improve the performance of
#    non-tree based methods. This could be made optional - one for future discussions.
# 6. Train a range of methods to relate peak height to features [point 3-4]. Train
#    a model for each m/z channel.
# 7. Generate stats on performance [R squared, Dot product, Absolute deviation in
#    height]. We will use the Dot product for comparing spectra and focus on specific
#    channels of relevance for the AMS.
# 8. (optional) plot average stats per m/z channel per model
# 9. (optional) plot overall model performance for all compounds [dot product boxplots]
##########################################################################################

##########################################################################################
# External library requirements:
# You will need to have the following available to import in Python:
# Scikit learn package
# Numpy / Scipy
# Matplotlib [for plots]
# UManSysProp [which also requires OpenBabel / Pybel to work]
#  - For this package clone our public release: https://github.com/loftytopping/UManSysProp_public
#  - Using the command: git clone https://github.com/loftytopping/UManSysProp_public.git
#
##########################################################################################

from os import listdir, getcwd
from os.path import isfile, join
import ipdb
import glob
import numpy
import csv
import scipy
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.svm import SVR
from sklearn import tree, grid_search
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, SGDRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import collections
import os.path
import sys
# Here you will need to put the relevant path in for your UManSysProp distribution 
sys.path.append('/home/david/Documents/Python/Git_repo_UManSysProp/umansysprop/')
from umansysprop import groups #umansysprop is constructed as a package
import collections
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
import keys_descriptors
import numpy
import EI_plotting
import pybel
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_regression

##########################################################################################
# 0. General model options
# --- Specify which fingerprint to use: ---
key_flag = 4
             # 1 - MACCS [generic fingerprint]
             # 2 - FP4 [generic fingerprint]
             # 3 - AIOMFAC [used for activity coefficient calculations]
             # 4 - Nanoolal [used in vapour pressure calculations]
      
# --- Use a subset for training and then evaluation ---
# Rather than a k-fold approach, given teh dataset is currently small, we decided
# to define a % of the total dataset ranked according to the number of features captured
# If and when more spectra are collected, this should be changed.      
subset_training=1 # Evaluate model performance by training to a subset                         
percent_training=80.0 # Define the % of that subset
            
# --- Use variable selection to use only relevant features in fitting ---
# This is based on a percentile of the highest scores             
variable_selection = 1 
percentile=30 # Define the percentile to use in variable selection
print_feature_importance = 0 #Print feature importance, and information, to file 

# --- Analyse feature importance ---
analyse_variable_selection = 0 # Even if not used in the fitting process, this option
                               # enables us to assess those features deemed most important
                               # per m/z channel. 
                               # Please note with the current script, this will result in 
                               # an error if AIOFMAC keys are used. 
                               # This is because I have yet to put together a 
                               # keys_descriptors file for AIOMFAC. 
                               # This will be added in a future release.
variable_selection_print = 1 # Print most informative features to a file                               
variable_selection_plot = 1 # Plot the feature importance on a 2D figure
variable_selection_plot_imp = 1 # Plot the feature importance of key channels on a 2D figure

# --- Define plotting /printing options ---
Visualise_keys = 1 # Generate a 2D plot showing sparsity of feature extraction
print_fullstats_permz = 0 # Print dot product stats per m/z channel
plot_fullstats_permz = 0 # Generate scatter plots per m/z channel over all compound
plot_fullstats_percomp = 1 # Generate boxplot showing dot product variability for full spectra
generate_spectra_plots = 1 # Generate individual spectra plots, measured and some models
stat_key_channel = 1 # Generate comparison statistics over key channels only

# --- Define m/z channels for which performance is evaluated ---
key_channels=[15,18,28,29,39,41,43,44,50,51,53,55,57,60,73,77,91]


##########################################################################################

# --- NEW SECTION --- 

##########################################################################################
# 1. Read in EI spectra, seperating only itx files from the rest
onlyfiles = [f for f in glob.glob('*.itx')]
#now create a matrix that will hold the EI spectra (raw)
EI_matrix=numpy.zeros((len(onlyfiles),450),)
# Now extract the data from each file and populate a matrix with all values, 
# saving the name to a file
step=0
filenames=[]
EI_name=[]
for filename in onlyfiles:
   SMILES_flag=0
   filenames.append(filename[:])
   text=open(filename[:],'rU')
   mz_step=0
   parsing = False
   for line in text.readlines():
       if line.startswith("END"):
           parsing = False
       if parsing == True:
           #extract numeric value if there is one, or parse a NaN
           input = line.split()
           if (mz_step==39):
               print input[0]
           if (input[0] != 'NAN') is True:
               try: 
                   EI_matrix[step,mz_step]=float(input[0])
                   mz_step=mz_step+1 
               except:
                   print "No entry"
           elif (input[0] == 'NAN') is True:
               EI_matrix[step,mz_step]=0
               mz_step=mz_step+1 
       # The itx files have key words which we use to start extracting data
       if line.startswith("BEGIN"):
           parsing = True
       if line.startswith("SMILES"):
           input = line.split()
           try:
               EI_name.append(input[1])
               SMILES_flag=1
           except:
               print "No SMILES"
   if (SMILES_flag==0):
      print "No SMILES for %s", filename
   step=step+1
##########################################################################################  

# --- NEW SECTION --- 

##########################################################################################
# 2. Remove channels with less than 1% total peak height or negative values
for i in range(len(onlyfiles)):
    for j in range(450):
       if (EI_matrix[i,j] < 0):
           EI_matrix[i,j]=0
   
for i in range(len(onlyfiles)):
    temp2=sum(EI_matrix[i,:])
    for j in range(450):
        EI_matrix[i,j]=EI_matrix[i,j]/temp2
    for j in range(450):
        if (EI_matrix[i,j])< 0.01:
            EI_matrix[i,j]=0
        
# Re-normalise the values to give a total of 100%
for i in range(len(onlyfiles)):
    temp2=sum(EI_matrix[i,:])
    for j in range(450):
        EI_matrix[i,j]=EI_matrix[i,j]/temp2
            
# Now lets pull out the m/z ratios that are non zero based, or have max value > 1% height 
zero_flag=numpy.zeros((450,1),)
for i in range(450):
    temp=numpy.count_nonzero(EI_matrix[:,i])
    if (max(EI_matrix[:,i]) < 0.01):
        zero_flag[i,0]=0.0
    elif (temp > 5) and (max(EI_matrix[:,i]) > 0.01):
        zero_flag[i,0]=1.0
EI_matrix_new=numpy.zeros((len(onlyfiles),sum(zero_flag),),)
mz_array=numpy.zeros((sum(zero_flag),1),)
step=0

for i in range(450):
    if (zero_flag[i,0] != 0.0):
        EI_matrix_new[:,step]=EI_matrix[:,i]
        mz_array[step,0]=i+1
        step=step+1
########################################################################################## 

# --- NEW SECTION ---       

##########################################################################################
# 3. Generate MACCS [or other] keys per compound and plot coverage (optional)
# Here we create some python dictionaries that assign arrays of features per compound
# This is before we convert the information into a matrix and then list for training
# Note all fingerprints are now stored in the UManSysProp package, rather than relying
# on RDKit and other packages. The provenance of the associated SMARTS strings is given
# in those UManSysProp files.

# All options use an ordered dict to store information on the keys extracted
key_dict = collections.OrderedDict()

if key_flag == 1: #MACCS keys 
    MACCS_keys = {} 
    key_dict_tag_maccs=[]
    for s in EI_name:
        SMILES=pybel.readstring(b'smi',s)
        MACCS_keys[s] = groups.MACCS(SMILES)
        key_dict.update({b : 1.0 for (b,v) in MACCS_keys[s].iteritems() if v > 0})
    key_dict_tag_maccs.append({b for (b,v) in key_dict.iteritems()}) 
elif key_flag == 2: #FP4 keys 
    FP4_keys = {}
    key_dict_tag_fp4=[]
    for s in EI_name:
        SMILES=pybel.readstring(b'smi',s)
        FP4_keys[s] = groups.FP4(SMILES)
        key_dict.update({b : 1.0 for (b,v) in FP4_keys[s].iteritems() if v > 0})
    key_dict_tag_fp4.append({b for (b,v) in key_dict.iteritems()}) 
elif key_flag == 3: #AIOMFAC keys 
    AIOMFAC_keys = {}
    key_dict_tag_aiomfac=[]
    for s in EI_name:
        SMILES=pybel.readstring(b'smi',s)
        AIOMFAC_keys[s] = groups.aiomfac(SMILES)
        #ipdb.set_trace()
        key_dict.update({b : 1.0 for (b,v) in AIOMFAC_keys[s].iteritems() if v > 0})
    key_dict_tag_aiomfac.append({b for (b,v) in key_dict.iteritems()}) 
elif key_flag == 4: #Nanoolal keys 
    Nanoolal_keys = {}
    key_dict_tag_Nanoolal=[]
    for s in EI_name:
        SMILES=pybel.readstring(b'smi',s)
        Nanoolal_keys[s] = groups.nannoolal_primary(SMILES)
        key_dict.update({b : 1.0 for (b,v) in Nanoolal_keys[s].iteritems() if v > 0})
    key_dict_tag_Nanoolal.append({b for (b,v) in key_dict.iteritems()}) 

# Whatever the option, we now need to convert this into a matrix and then list for training
# This also means we can visualise the sparsity of the feature coverage

Key_matrix=numpy.zeros((len(onlyfiles),len(key_dict)),)
Key_flag=numpy.zeros((len(onlyfiles),))

# Since we keep the names in a list, this will preserve order of input
# Note here we need to decide whether we keep the full sotchiometry information
# or simply register the presence of a group with 1.0 in the key_matrix. The
# idea behind this is derived from how MACCS and FP4 keys are reported. Since we have
# the individual SMARTS we have the option of retaining stochiometry. This could
# be changed quite easily as a modular option if required.

step1=0
step2=0
X_flag_key={}
nonzero=0
for s in EI_name:
    step2=0
    if key_flag == 1:
        for b in key_dict_tag_maccs[0]:
            for (b2,v2) in MACCS_keys[s].iteritems():
                if b2==b:
                     if v2 > 0:
                        Key_matrix[step1, step2]= v2 # Change to 1 if simply want to identify feature presence
            step2=step2+1
    elif key_flag == 2:
        for b in key_dict_tag_fp4[0]:
            for (b2,v2) in FP4_keys[s].iteritems():
                if b2==b:
                    if v2 > 0:
                        Key_matrix[step1, step2]= v2 # Change to 1 if simply want to identify feature presence
            step2=step2+1
    elif key_flag == 3:
        for b in key_dict_tag_aiomfac[0]:
            for (b2,v2) in AIOMFAC_keys[s].iteritems():
                if b2==b:
                    if v2 > 0:
                        Key_matrix[step1, step2]= v2 # Change to 1 if simply want to identify feature presence
            step2=step2+1                                                
    elif key_flag == 4:
        for b in key_dict_tag_Nanoolal[0]:
            for (b2,v2) in Nanoolal_keys[s].iteritems():
                if b2==b:
                    if v2 > 0:
                        Key_matrix[step1, step2]= v2 # Change to 1 if simply want to identify feature presence                                            
            step2=step2+1
    nonzero_new=numpy.count_nonzero(Key_matrix[step1, :])
    if nonzero_new > nonzero:
        nonzero=nonzero_new
    step1=step1+1

# Normalise the key matrix using the min_max_scalar. Currently use this option buy default.
temp_array1=numpy.asarray(Key_matrix)
##X_scaled = preprocessing.scale(temp_array1)
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(temp_array1)
Key_matrix2=numpy.matrix(X_scaled)

print "Maximum features for any given compounds = ",nonzero
    
# ---------- Optional plotting ---------------
# Send the matrix to a function for visualisation
labels=['MACCS keys','FP4 keys','AIOMFAC keys','Nanoolal keys']
if Visualise_keys == 1:
    EI_plotting.Visualise_keys(Key_matrix2)
# --------------------------------------------

# -----------Define a subset for training -------------
# Rather than use a random selection, we currently use those compounds with the 
# highest number of features and then define a % of the total to use in the training
# process.

if subset_training == 1:
    subset_keys=numpy.argsort(numpy.sum(Key_matrix, axis=1))[::-1][0:int(round(len(EI_name)*percent_training/100.0))]
else:
    subset_keys=[None]*len(EI_name) 
    for step in range(len(subset_keys)):
        subset_keys[step]=step

##########################################################################################     

# --- NEW SECTION ---   

##########################################################################################
# 4. Variable selection analysis (optional) - output information on those features deemed
#    important to predict peak heights. This is based on a set percentile of top x%.
#    This can be expanded in the future and should be made into a module

if analyse_variable_selection == 1:
    
    feature_importance=numpy.zeros((len(mz_array),len(key_dict)),) # array to save importance values
    feature_importance_num=numpy.zeros((len(mz_array),2),)
    Important_keys=collections.defaultdict(lambda: collections.defaultdict()) # dict to record the most important keys
    Important_keys_tag=collections.defaultdict(lambda: collections.defaultdict()) # dict to turn that information into strings
                                                                                  # according to information held in 'keys_descriptors.py'
    Specific_keys_features=collections.defaultdict(lambda: collections.defaultdict())

    # extract the dictionaries holding the descriptors
    info_maccs=keys_descriptors.MACCS_tags()
    info_fp4=keys_descriptors.FP4_tags()
    info_Nanoolal=keys_descriptors.Nanoolal_P_tags()
    if key_flag == 1:
        info=info_maccs
    elif key_flag == 2:
        info=info_fp4
    elif key_flag == 4:
        info=info_Nanoolal

    # Now cycle through each m/z channel and extract the relevant features
    for mz_step in range(len(mz_array)):
        Y_response=numpy.zeros((len (onlyfiles)),) #This has one m/z value per run and holds the 
                                                #normalised peak height information
        non_neg_values=numpy.zeros((len (onlyfiles)),) #This has one m/z value per run
        step1=0
        occurance=0
        location=[]
        max_group_num1=0
        for s in EI_name:
            Y_response[step1]=EI_matrix_new[step1,mz_step]
            max_group_num2=numpy.count_nonzero(Key_matrix[step1, :])
            if max_group_num2>max_group_num1:
                max_group_num1=max_group_num2
            if Y_response[step1]>0:
                non_neg_values[step1]=1.0
                occurance=occurance+1
                location.append(step1)
            step1=step1+1
            occurance=occurance/len(EI_name)

        feature_importance_num[mz_step,1]=max_group_num1
        # now find out how many of the compounds are covered in this m/z
        number=numpy.count_nonzero(Y_response)
        # Convert arrays to list for the fitting procedure
        Key_matrix_list=Key_matrix2.tolist()
        Y_response_list=Y_response.tolist()
        
        #Now try to use univariate feature selection
        selector = SelectPercentile(f_regression, percentile)
        selector.fit(Key_matrix_list, Y_response_list)
        scores = -numpy.log10(selector.pvalues_)
        scores /= scores.max()
        importances = scores
        
        indices = numpy.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        
        # Also save this information to a matrix, for each m/z channel. When we have this, we can
        # plot the average 'importance' per feature and see where the biggest contributors are
        for f in range(Key_matrix.shape[1]):
            if importances[indices[f]] > 0.0: # this is currently arbitrary - need to check this
                feature_importance[mz_step,indices[f]]=importances[indices[f]]
                #Now create a dictionary that holds the sorted features that are picked as the most important
                Important_keys[mz_step][info[key_dict.items()[indices[f]][0]]]=importances[indices[f]]  
                #pdb.set_trace()
                Important_keys_tag[mz_step][indices[f]]=info[key_dict.items()[indices[f]][0]]
                feature_importance_num[mz_step,0]+=1
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    for channel in key_channels:
        for tag in Important_keys[numpy.where(mz_array == channel)[0][0]]:
            Specific_keys_features[channel][tag]=Important_keys[numpy.where(mz_array == channel)[0][0]][tag]
    
    #4a (optional) - Store the feature importance to a file
    if variable_selection_print == 1:
        #Going to record the feature importance per channel and plot on a surface
        #We can then see which key channels are dominated by a certain number of features, or not
        #Then make these channels clear on the graph - we already have this information but need to pull
        #out the key channels
        f = open('Final_Key_features.txt','w')
        for channel in key_channels:
            temp=sorted(Specific_keys_features[channel], key=Specific_keys_features[channel].get, reverse=True)
            for tag in temp:
                f.write('\t '+tag) 
                f.write('\t '+str(Specific_keys_features[channel][tag])) 
            f.write('\n')
        f.close()
    #4b (optional) - Plot the feature importance on a 2D figure
    if variable_selection_plot == 1:
        EI_plotting.Feature_importance_gen(feature_importance)
        if variable_selection_plot_imp == 1:
            feature_importance_key_channels=numpy.zeros((len(key_channels),len(key_dict)),)
            step=0
            for channel in key_channels:
                for index in range(len(key_dict)):
                    feature_importance_key_channels[step,index]=feature_importance[numpy.where(mz_array == channel)[0][0],index]
                step+=1
            EI_plotting.Feature_importance_spec(feature_importance_key_channels,key_channels)


##########################################################################################   

# --- NEW SECTION ---     

##########################################################################################
# 5. Train a model to predict the occurance of m/z channels (non-optional).              
# Fit a model to predict if an m/z channel will occur or not. This is then used before   
# fitting peak height dependency. The idea here is that it might make    
# the performance on non-tree based algorithms better                                    
# We use an ensemble classifier for this purpose. This can be changed                  
total_deviation_ensemble_tree_matrix_mzselect=numpy.zeros((len(mz_array),3),) # array to hold stats on performance
predicted_mzselect=numpy.zeros((len(onlyfiles),len (mz_array)),) # array to hold predicted mz occurance 
forest = ExtraTreesRegressor()
for mz_step in range(len(mz_array)):
    Y_mz_presence=numpy.zeros((len (onlyfiles)),) #This has one m/z value per run
    step1=0
    for s in EI_name:
        if (EI_matrix_new[step1,mz_step]>0):
            Y_mz_presence[step1]=1.0
        step1=step1+1
    Key_matrix_list=Key_matrix.tolist()
    Key_matrix_train=Key_matrix[subset_keys].tolist()
    Y_mz_presence_list=Y_mz_presence[subset_keys].tolist()
    y_tree_ensemble=forest.fit(Key_matrix_train, Y_mz_presence_list).predict(Key_matrix_list)
    total_deviation_ensemble_tree_matrix_mzselect[mz_step,0]=numpy.nanmean(abs(Y_mz_presence-y_tree_ensemble))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_mz_presence,y_tree_ensemble)
    total_deviation_ensemble_tree_matrix_mzselect[mz_step,1]=r_value**2.0
    step1=0
    for s in EI_name:
        predicted_mzselect[step1,mz_step]=y_tree_ensemble[step1]
        step1+=1

# This matrix of predicted mz occurances will be used to select which channel to
# predict the peak height for, and also used in deriving performance statistics
##########################################################################################

# --- NEW SECTION ---

##########################################################################################
# 6. Train a range of methods to relate peak height to features [point 3-4]. Train       
#    a model for each m/z channel.                   

# We use the following algorithms from the Regression package of Scikit learn:
# : http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
# - Decision tree
# - Support Vector Machines (3 different kernels)
# - Bayesian Ridge
# - Ordinary Least Squares
# - Stochastic Gradient Descent

# First we need to initialise a range of variables that take the model predictions and 
# generate peformance statistics. To make this tidy these could be imported as global
# variables and held in a seperate module, but I have listed them below for now. Note
# that variables are global to the module they are defined in, so if this is changed,
# you will need to use explicit global declarations. If you extend this package to use
# other algorithms, you will need to create another entry below, using the same format.

# Arrays to hold performance statistics:
total_deviation_tree_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_ensemble_tree_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_brr_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_ols_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_sgdr_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_rbf_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_poly_matrix=numpy.zeros((len(mz_array),3),)
total_deviation_lin_matrix=numpy.zeros((len(mz_array),3),)
# Dictionaries to hold m/z peak height predictions
mz_pred_rbf_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_lin_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_poly_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_brr_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_ols_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_sgdr_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_tree_matrix=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_ensemble_tree_matrix=collections.defaultdict(lambda: collections.defaultdict())
# Arrays to hold performance statistics for full spectra predictions for a given compound
total_deviation_rbf_full=numpy.zeros((len (onlyfiles)),)
total_deviation_poly_full=numpy.zeros((len (onlyfiles)),)
total_deviation_lin_full=numpy.zeros((len (onlyfiles)),)
total_deviation_brr_full=numpy.zeros((len (onlyfiles)),)
total_deviation_ols_full=numpy.zeros((len (onlyfiles)),)
total_deviation_sgdr_full=numpy.zeros((len (onlyfiles)),)
total_deviation_tree_full=numpy.zeros((len (onlyfiles)),)
total_deviation_ensemble_tree_full=numpy.zeros((len (onlyfiles)),)
# Dictionaries to hold m/z peak height predictions
mz_pred_rbf_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_lin_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_poly_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_brr_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_ols_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_sgdr_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_tree_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())
mz_pred_ensemble_tree_matrix_nosub=collections.defaultdict(lambda: collections.defaultdict())

# Now cycle through each m/z channel and fit a model [using the previous predictor of
# whether the m/z channel has data in it for evaluating performance]

key_location=[]
size_array=numpy.zeros((len (mz_array)),)

for mz_step in range(len(mz_array)):

    # Check to see if this m/z channel is in our list of ones to watch
    if mz_array[mz_step] in key_channels:
        key_location.append(mz_step)
    
    print "mz_step",mz_step # Keep user informed of progress
    Y_peak_height=numpy.zeros((len (onlyfiles)),) #This has one m/z value per run
    non_neg_values=numpy.zeros((len (onlyfiles)),) #This has one m/z value per run
    step1=0
    occurance=0
    location=[]
    location_mzpredict=[]
    for s in EI_name:
        Y_peak_height[step1]=EI_matrix_new[step1,mz_step]
        #Here we can extract the location of whether the ensemble tree for predicting the non-negative nature of m/z
        #says there should be a value
        if predicted_mzselect[step1,mz_step]>0:
           location_mzpredict.append(step1)        
        if step1 in subset_keys:
           non_neg_values[step1]=1.0
           occurance=occurance+1
           location.append(step1)
        step1=step1+1
        occurance=occurance/len(EI_name)

    #now find out how many of the compounds are covered in this m/z
    number=numpy.count_nonzero(Y_peak_height)
    size_array[mz_step]=number

    if variable_selection == 1 and sum(Y_peak_height[location])>0.0:
        
        #recalculate feature importance and then transform the fitting variable
        selector = SelectPercentile(f_regression, percentile)
        selector.fit(Key_matrix2[location].tolist(), Y_peak_height[location].tolist())
        scores = -numpy.log10(selector.pvalues_)
        scores /= scores.max()
        importances = scores
        
        Key_matrix_training_list=selector.transform(Key_matrix2[location].tolist()) 
        Key_matrix_list=selector.transform(Key_matrix2.tolist())      
        
    else:
        Key_matrix_training_list=Key_matrix2[location].tolist()
        Key_matrix_list=Key_matrix2.tolist()        

    Y_peak_height_list=Y_peak_height.tolist()

    #just keeping the training sample the same as the original [using only non zero values]
    Y_peak_height_training_list=Y_peak_height[location].tolist()
    
    #Now start the training. For some algorithms there are multiple tuning parameters. We cycle through
    #those where appropriate, using R squared stats to decide the optimum combination. 

    #6.1 --- Support Vector Machine ---
    # Here we fit 3 different SVM kernels to the data
    # Define parameters used in the SVM kernels
    C_array= [1e-3,1e-2,1e-1,1,1.0e1, 1.0e2, 1.0e3,1.0e4,1.0e8,1.0e10]
    gamma_array=[1.0e-5,1.0e-3,1.0e-2,0.1]
    degree_array=[3,4,5,6,8,9,10]
    R_squared_old1=-2
    R_squared_old2=-2
    R_squared_old3=-2
    for Cd in C_array:
        for gammad in gamma_array:
            for degreed in degree_array:
                ##################RBF Kernel################
                # Set up the rbf kernel
                svr_rbf_test = SVR(kernel='rbf', gamma=gammad,degree=degreed,C=Cd,tol=1.0e-6,max_iter=5000)  
                #if mz_step==23:
                #    ipdb.set_trace()            
                y_rbf_test = svr_rbf_test.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
                # I'm going to remove any predictions that have a peak height less than 0.01 and also compare stats across all values, not just non-zero
                numpy.putmask(y_rbf_test, y_rbf_test<0.01, 0.0)
                # Generate stats of predicted versus actual data - 
                new_list=y_rbf_test
                if len(location_mzpredict)>0:
                    for i in range(len(Y_peak_height)):
                        if i not in location_mzpredict:
                            new_list[i]=0.0
                else:
                    new_list[:]=0.0
                
		slope, intercept, r_value, p_value, std_err = stats.linregress(new_list,Y_peak_height)
		    
                R_squared=r_value**2.0
                if R_squared > R_squared_old1:
                    R_squared_old1=R_squared
                    y_rbf_best_params=[Cd,gammad,degreed]
                    y_rbf_best=new_list
    # Now derive some performance stats for all SVM kernels.
    # Column 0 - absolute deviation in peak height
    # Column 1 - R squared
    # Column 2 - Dot product (Cosine)
    total_deviation_rbf_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_rbf_best))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_rbf_best)
    total_deviation_rbf_matrix[mz_step,1]=r_value**2.0
    total_deviation_rbf_matrix[mz_step,2]=numpy.dot(y_rbf_best,Y_peak_height)/(numpy.linalg.norm(y_rbf_best)*numpy.linalg.norm(Y_peak_height))
                    
    for Cd in C_array:
        for gammad in gamma_array:
            for degreed in degree_array:
                #################Poly Kernel################
                svr_poly_test = SVR(kernel='poly',gamma=gammad,degree=degreed,C=Cd,tol=1.0e-6,max_iter=5000)
                y_poly_test = svr_poly_test.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
                numpy.putmask(y_poly_test, y_poly_test<0.01, 0.0)
                new_list=y_poly_test
                if len(location_mzpredict)>0:
                    for i in range(len(Y_peak_height)):
                        if i not in location_mzpredict:
                            new_list[i]=0.0
                else:
                    new_list[:]=0.0

		slope, intercept, r_value, p_value, std_err = stats.linregress(new_list,Y_peak_height)
		R_squared=r_value**2.0
                #total_deviation2=sum(abs(Y_MACCS_scaled_list/y_poly_test))
                if R_squared > R_squared_old2:
                    R_squared_old1=R_squared
                    y_poly_best_params=[Cd,gammad,degreed]
                    y_poly_best=new_list
                    
    # Now derive some performance stats for all SVM kernels.
    # Column 0 - absolute deviation in peak height
    # Column 1 - R squared
    # Column 2 - Dot product (Cosine)
    total_deviation_poly_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_poly_best))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_poly_best)
    total_deviation_poly_matrix[mz_step,1]=r_value**2.0
    total_deviation_poly_matrix[mz_step,2]=numpy.dot(y_poly_best,Y_peak_height)/(numpy.linalg.norm(y_poly_best)*numpy.linalg.norm(Y_peak_height))
                    
    for Cd in C_array:                    
        #################Linear Kernel################
        # Only one tunable parameter
        svr_lin_test = SVR(kernel='linear',C=Cd,max_iter=1000)
        y_lin_test = svr_lin_test.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
        numpy.putmask(y_lin_test, y_lin_test<0.01, 0.0)
        new_list=y_lin_test
        if len(location_mzpredict)>0:
            for i in range(len(Y_peak_height)):
                if i not in location_mzpredict:
                    new_list[i]=0.0
        else:
            new_list[:]=0.0
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_list,Y_peak_height)

	#slope, intercept, r_value, p_value, std_err = stats.linregress(y_lin_test[location_mzpredict],Y_peak_height[location_mzpredict])
        if R_squared > R_squared_old3:
            R_squared_old3=R_squared
            y_lin_best_params=[Cd]
            y_lin_best=new_list

    # Now derive some performance stats for all SVM kernels.
    # Column 0 - absolute deviation in peak height
    # Column 1 - R squared
    # Column 2 - Dot product (Cosine)
    total_deviation_lin_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_lin_best))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_lin_best)
    total_deviation_lin_matrix[mz_step,1]=r_value**2.0
    total_deviation_lin_matrix[mz_step,2]=numpy.dot(y_lin_best,Y_peak_height)/(numpy.linalg.norm(y_lin_best)*numpy.linalg.norm(Y_peak_height))

    #6.2 --- Decision tree ---
    # Decision tree model - cycling through tree maximum depth
    # Replicate same procedure as above, cycle through until best fit
    tree_depth_array=[2,4,6,8,10,12,20]
    total_deviation_tree_old=1.0e8
    R_tree_old=-2
    average_deviation_tree=0
    for tree_depth in tree_depth_array:
        clf=DecisionTreeRegressor(max_depth=tree_depth)  
	y_tree_all = clf.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
	numpy.putmask(y_tree_all, y_tree_all<0.01, 0.0) # remove data below 0.01 as not used
	new_list=y_tree_all
        if len(location_mzpredict)>0:
            for i in range(len(Y_peak_height)):
                if i not in location_mzpredict:
                    new_list[i]=0.0
        else:
            new_list[:]=0.0
        
	slope, intercept, r_value, p_value, std_err = stats.linregress(new_list,Y_peak_height)
	R_tree=r_value**2.0
        if R_tree > R_tree_old:
            R_tree_old=R_tree
            y_tree_best_params=[tree_depth]
            y_tree_best=new_list
    # Generate performance statistics
    total_deviation_tree_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_tree_best))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_tree_best)
    total_deviation_tree_matrix[mz_step,1]=r_value**2.0
    total_deviation_tree_matrix[mz_step,2]=numpy.dot(y_tree_best,Y_peak_height)/(numpy.linalg.norm(y_tree_best)*numpy.linalg.norm(Y_peak_height))

    #6.3 --- Ensemble forest ---
    # Unlike the decision tree, the parameters are optimised internally.
    y_tree_ensemble=forest.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
    numpy.putmask(y_tree_ensemble, y_tree_ensemble<0.01, 0.0)
    new_list=y_tree_ensemble
    if len(location_mzpredict)>0:
        for i in range(len(Y_peak_height)):
            if i not in location_mzpredict:
                new_list[i]=0.0
    else:
        new_list[:]=0.0
    y_tree_ensemble=new_list    
    
    # Generate performance statistics    
    total_deviation_ensemble_tree_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_tree_ensemble))
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_tree_ensemble)
    total_deviation_ensemble_tree_matrix[mz_step,1]=r_value**2.0
    total_deviation_ensemble_tree_matrix[mz_step,2]=numpy.dot(y_tree_ensemble,Y_peak_height)/(numpy.linalg.norm(y_tree_ensemble)*numpy.linalg.norm(Y_peak_height))
    
    #6.4 --- Bayesian Ridge ---
    brr = BayesianRidge(compute_score=True)
    y_brr= brr.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
    numpy.putmask(y_brr,y_brr<0.01, 0.0)
    new_list=y_brr
    if len(location_mzpredict)>0:
        for i in range(len(Y_peak_height)):
            if i not in location_mzpredict:
                new_list[i]=0.0
    else:
        new_list[:]=0.0
    y_brr=new_list    
    # Generate performance statistics       
    total_deviation_brr_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_brr)) 
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_brr)
    total_deviation_brr_matrix[mz_step,1]=r_value**2.0
    total_deviation_brr_matrix[mz_step,2]=numpy.dot(y_brr,Y_peak_height)/(numpy.linalg.norm(y_brr)*numpy.linalg.norm(Y_peak_height))

    #6.5 --- Ordinary Least Squared --- 
    ols = LinearRegression()
    y_ols= ols.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
    numpy.putmask(y_ols,y_ols<0.01, 0.0)  
    new_list=y_ols
    if len(location_mzpredict)>0:
        for i in range(len(Y_peak_height)):
            if i not in location_mzpredict:
                new_list[i]=0.0
    else:
        new_list[:]=0.0
    y_ols=new_list         
    # Generate performance statistics               
    total_deviation_ols_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_ols))  
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_ols)
    total_deviation_ols_matrix[mz_step,1]=r_value**2.0
    total_deviation_ols_matrix[mz_step,2]=numpy.dot(y_ols,Y_peak_height)/(numpy.linalg.norm(y_ols)*numpy.linalg.norm(Y_peak_height))

    #6.6 --- Stochastic Gradient Descent --- 
    SGDR = SGDRegressor()
    y_sgdr= SGDR.fit(Key_matrix_training_list, Y_peak_height_training_list).predict(Key_matrix_list)
    numpy.putmask(y_sgdr, y_sgdr<0.01, 0.0)
    new_list=y_sgdr
    if len(location_mzpredict)>0:
        for i in range(len(Y_peak_height)):
            if i not in location_mzpredict:
                new_list[i]=0.0
    else:
        new_list[:]=0.0
    y_sgdr=new_list             
    # Generate performance statistics                   
    total_deviation_sgdr_matrix[mz_step,0]=numpy.nanmean(abs(Y_peak_height-y_sgdr))     
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_peak_height,y_sgdr)
    total_deviation_sgdr_matrix[mz_step,1]=r_value**2.0
    total_deviation_sgdr_matrix[mz_step,2]=numpy.dot(y_sgdr,Y_peak_height)/(numpy.linalg.norm(y_sgdr)*numpy.linalg.norm(Y_peak_height))

    ###############################################
    # Store the predicted height per m/z channel for spectra-spectra comparison
    # Whilst we test the performance of invidiaul m/z bsaed predictors, we also
    # need to compare the performance of entire spectra
    # We need to decide, however, whether we just use our 'key channels' or the
    # entire spectrum
    name_step=0
    # We should also save results of compounds not in the training subset 
    for s in EI_name:
        #print "s=",s
        mz_pred_rbf_matrix[s][mz_array[mz_step][0]]=y_rbf_best[name_step]
	mz_pred_lin_matrix[s][mz_array[mz_step][0]]=y_lin_best[name_step]
	mz_pred_poly_matrix[s][mz_array[mz_step][0]]=y_poly_best[name_step]
        mz_pred_brr_matrix[s][mz_array[mz_step][0]]=y_brr[name_step]
	mz_pred_ols_matrix[s][mz_array[mz_step][0]]=y_ols[name_step]
	mz_pred_sgdr_matrix[s][mz_array[mz_step][0]]=y_sgdr[name_step]
	mz_pred_tree_matrix[s][mz_array[mz_step][0]]=y_tree_best[name_step]
	mz_pred_ensemble_tree_matrix[s][mz_array[mz_step][0]]=y_tree_ensemble[name_step]
	
	if step1 not in subset_keys:
            mz_pred_rbf_matrix_nosub[s][mz_array[mz_step][0]]=y_rbf_best[name_step]
	    mz_pred_lin_matrix_nosub[s][mz_array[mz_step][0]]=y_lin_best[name_step]
	    mz_pred_poly_matrix_nosub[s][mz_array[mz_step][0]]=y_poly_best[name_step]
            mz_pred_brr_matrix_nosub[s][mz_array[mz_step][0]]=y_brr[name_step]
	    mz_pred_ols_matrix_nosub[s][mz_array[mz_step][0]]=y_ols[name_step]
	    mz_pred_sgdr_matrix_nosub[s][mz_array[mz_step][0]]=y_sgdr[name_step]
	    mz_pred_tree_matrix_nosub[s][mz_array[mz_step][0]]=y_tree_best[name_step]
	    mz_pred_ensemble_tree_matrix_nosub[s][mz_array[mz_step][0]]=y_tree_ensemble[name_step]
	    	
		
        name_step=name_step+1
        
        
##########################################################################################

# --- NEW SECTION ---

##########################################################################################
# 7 - Generate full spectra comparison statistics

# all out.
step1=0
temp_step=0
for s in EI_name:
    
    # Initialise arrays that hold peak heights for all m/z channels for one compound
    y_rbf=numpy.zeros((len(mz_array)),)
    y_lin=numpy.zeros((len(mz_array)),)
    y_poly=numpy.zeros((len(mz_array)),)
    y_brr=numpy.zeros((len(mz_array)),)
    y_ols=numpy.zeros((len(mz_array)),)
    y_sgdr=numpy.zeros((len(mz_array)),)
    y_tree=numpy.zeros((len(mz_array)),)
    y_ensemble_tree=numpy.zeros((len(mz_array)),)
    
    # Store measured data
    Y_peak_height=numpy.zeros((len (mz_array)),) 
    
    # Only select m/z data that has a non-zero value
    location_mzpredict=[]
    
    # Now extract the data from the full dictionary of values
    for mz_step2 in range(len(mz_array)):
        Y_peak_height[mz_step2]=EI_matrix_new[step1,mz_step2]
        y_rbf[mz_step2]=mz_pred_rbf_matrix[s][mz_array[mz_step2][0]]
        y_lin[mz_step2]=mz_pred_lin_matrix[s][mz_array[mz_step2][0]]
        y_poly[mz_step2]=mz_pred_poly_matrix[s][mz_array[mz_step2][0]]
        y_brr[mz_step2]=mz_pred_brr_matrix[s][mz_array[mz_step2][0]]
	y_ols[mz_step2]=mz_pred_ols_matrix[s][mz_array[mz_step2][0]]
	y_sgdr[mz_step2]=mz_pred_sgdr_matrix[s][mz_array[mz_step2][0]]
	y_tree[mz_step2]=mz_pred_tree_matrix[s][mz_array[mz_step2][0]]
	y_ensemble_tree[mz_step2]=mz_pred_ensemble_tree_matrix[s][mz_array[mz_step2][0]]
	if predicted_mzselect[step1,mz_step2]>0:
            location_mzpredict.append(mz_step2)     
            
    # Here we have options according to whether we compare complete spectra, or only those from key channels
    
    if stat_key_channel == 1:  

        total_deviation_rbf_full[step1]=numpy.dot(y_rbf[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_rbf[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_poly_full[step1]=numpy.dot(y_poly[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_poly[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_lin_full[step1]=numpy.dot(y_lin[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_lin[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))    
        total_deviation_brr_full[step1]=numpy.dot(y_brr[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_brr[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_ols_full[step1]=numpy.dot(y_ols[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_ols[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_sgdr_full[step1]=numpy.dot(y_sgdr[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_sgdr[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_tree_full[step1]=numpy.dot(y_tree[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_tree[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))
        total_deviation_ensemble_tree_full[step1]=numpy.dot(y_ensemble_tree[key_location],Y_peak_height[key_location])/(numpy.linalg.norm(y_ensemble_tree[key_location])*numpy.linalg.norm(Y_peak_height[key_location]))

    else:

        total_deviation_rbf_full[step1]=numpy.dot(y_rbf,Y_peak_height)/(numpy.linalg.norm(y_rbf)*numpy.linalg.norm(Y_peak_height))
        total_deviation_poly_full[step1]=numpy.dot(y_poly,Y_peak_height)/(numpy.linalg.norm(y_poly)*numpy.linalg.norm(Y_peak_height))
        total_deviation_lin_full[step1]=numpy.dot(y_lin,Y_peak_height)/(numpy.linalg.norm(y_lin)*numpy.linalg.norm(Y_peak_height))    
        total_deviation_brr_full[step1]=numpy.dot(y_brr,Y_peak_height)/(numpy.linalg.norm(y_brr)*numpy.linalg.norm(Y_peak_height))
        total_deviation_ols_full[step1]=numpy.dot(y_ols,Y_peak_height)/(numpy.linalg.norm(y_ols)*numpy.linalg.norm(Y_peak_height))
        total_deviation_sgdr_full[step1]=numpy.dot(y_sgdr,Y_peak_height)/(numpy.linalg.norm(y_sgdr)*numpy.linalg.norm(Y_peak_height))
        total_deviation_tree_full[step1]=numpy.dot(y_tree,Y_peak_height)/(numpy.linalg.norm(y_tree)*numpy.linalg.norm(Y_peak_height))
        total_deviation_ensemble_tree_full[step1]=numpy.dot(y_ensemble_tree,Y_peak_height)/(numpy.linalg.norm(y_ensemble_tree)*numpy.linalg.norm(Y_peak_height))

    step1=step1+1

##########################################################################################

# --- NEW SECTION ---

##########################################################################################
# 8 - Generate general plots and statistics of performance

# Generate boxplots for key m/z channels across all compounds
if plot_fullstats_percomp == 1:
    EI_plotting.Boxplot_all(total_deviation_rbf_full,total_deviation_poly_full,total_deviation_lin_full,total_deviation_brr_full,
    total_deviation_ols_full,total_deviation_sgdr_full,total_deviation_tree_full,total_deviation_ensemble_tree_full)

# Write the results to a file
f = open('Final_Statistics.txt','w')
f.write('Method'+'\t '+'Lower quartile'+'\t '+'Median'+'\t '+'Upper quartile') 
f.write('\n')
f.write('SVM RBF'+'\t '+str(numpy.nanpercentile(total_deviation_rbf_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_rbf_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_rbf_full, 75))) 
f.write('\n')
f.write('SVM Poly'+'\t '+str(numpy.nanpercentile(total_deviation_poly_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_poly_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_poly_full, 75)))  
f.write('\n')
f.write('SVM Lin'+'\t '+str(numpy.nanpercentile(total_deviation_lin_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_lin_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_lin_full, 75))) 
f.write('\n')
f.write('BRR'+'\t '+str(numpy.nanpercentile(total_deviation_brr_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_brr_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_brr_full, 75))) 
f.write('\n')
f.write('OLS'+'\t '+str(numpy.nanpercentile(total_deviation_ols_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_ols_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_ols_full, 75))) 
f.write('\n')
f.write('SGDR'+'\t '+str(numpy.nanpercentile(total_deviation_sgdr_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_sgdr_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_sgdr_full, 75))) 
f.write('\n')
f.write('Tree'+'\t '+str(numpy.nanpercentile(total_deviation_tree_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_tree_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_tree_full, 75))) 
f.write('\n')
f.write('Forest'+'\t '+str(numpy.nanpercentile(total_deviation_ensemble_tree_full, 25))+'\t '+str(numpy.nanpercentile(total_deviation_ensemble_tree_full, 50))+'\t '+str(numpy.nanpercentile(total_deviation_ensemble_tree_full, 75))) 
f.write('\n')
f.close()

# Plot the data for compounds not included in the training subset
# and save that data into a file for using in other packages
# This will only treat compounds not in a training subset. Therefore
# if you modify that selection, this will change.

file_name = open('Spectra_comparisons.txt','w')

step1=0
for s in EI_name:

    if step1 not in subset_keys:
    
        y_rbf=numpy.zeros((len(mz_array)),)
        y_brr=numpy.zeros((len(mz_array)),)
        y_ols=numpy.zeros((len(mz_array)),)
        y_tree=numpy.zeros((len(mz_array)),)
        y_ensemble_tree=numpy.zeros((len(mz_array)),)
        y_sgdr=numpy.zeros((len(mz_array)),)
        y_lin=numpy.zeros((len(mz_array)),)
        y_poly=numpy.zeros((len(mz_array)),)
        
        location_mzpredict=[]
        
        for mz_step2 in range(len(mz_array)):
            y_rbf[mz_step2]=mz_pred_rbf_matrix[s][mz_array[mz_step2][0]]
            y_poly[mz_step2]=mz_pred_poly_matrix[s][mz_array[mz_step2][0]]
            y_brr[mz_step2]=mz_pred_brr_matrix[s][mz_array[mz_step2][0]]
            y_ols[mz_step2]=mz_pred_ols_matrix[s][mz_array[mz_step2][0]]
            y_tree[mz_step2]=mz_pred_tree_matrix[s][mz_array[mz_step2][0]]
            y_ensemble_tree[mz_step2]=mz_pred_ensemble_tree_matrix[s][mz_array[mz_step2][0]]
            y_sgdr[mz_step2]=mz_pred_sgdr_matrix[s][mz_array[mz_step2][0]]
            y_lin[mz_step2]=mz_pred_lin_matrix[s][mz_array[mz_step2][0]]
             
            if predicted_mzselect[step1,mz_step2]>0:
                location_mzpredict.append(mz_step2)            
        
        #Save the experimental data in a file
        file_name.write('\t '+str(s)+'\t'+str(step1)) 
        file_name.write('\n')
        temp_step=0
        for mz in mz_array:
            file_name.write('\t '+str(mz[0]))
            temp_step+=1
        temp_step=0
        file_name.write('\n')
        for mz in mz_array:
            file_name.write('\t '+str(EI_matrix_new[step1,temp_step]))
            temp_step+=1
        file_name.write('\n')
        
        #Now save the model predictions to a file - Decision Tree in this instance
        file_name.write('\t'+'Tree method') 
        file_name.write('\n')
        file_name.write('\t'+str(numpy.round(total_deviation_tree_full[step1],decimals=4))) 
        file_name.write('\n')
        for key in location_mzpredict:
            file_name.write('\t '+str(mz_array[key][0]))
        file_name.write('\n')
        temp_step=0
        file_name.write('\n')
        #we need to normalise all values to equal 1..hence derive a factor
        temp_num=sum(y_tree[location_mzpredict])
        factor=1.0/temp_num
        for key in location_mzpredict:
            file_name.write('\t '+str(y_tree[key]*factor))
            temp_step+=1
        file_name.write('\n')       
            
        # row and column sharing
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharex='col', sharey='row')
        
        # Tree
        #ax1.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax1.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax1.plot(mz_array[location_mzpredict], y_tree[location_mzpredict], c='y', label='Tree')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax1.stem(mz_array[location_mzpredict]+0.5, y_tree[location_mzpredict], label='Tree')
        else:
            markerline, stemlines, baseline = ax1.stem(mz_array, [0.0]*len(mz_array), label='Tree')
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax1.set_title(numpy.round(total_deviation_tree_full[step1],decimals=4))
        ax1.legend()
        ax1.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax1.set_xlim(0, 80)
        
        # Ensemble tree
        #ax2.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax2.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax2.plot(mz_array[location_mzpredict], y_ensemble_tree[location_mzpredict], c='y', label='Ens tree')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax2.stem(mz_array[location_mzpredict]+0.5, y_ensemble_tree[location_mzpredict], label='Ens Tree')
        else:
            markerline, stemlines, baseline = ax2.stem(mz_array, [0.0]*len(mz_array), label='Ens Tree')    
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax2.set_title(numpy.round(total_deviation_ensemble_tree_full[step1],decimals=4))
        ax2.legend()
        ax2.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax2.set_xlim(0, 80)
        
        # RBF
        #ax3.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax3.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax3.plot(mz_array[location_mzpredict], y_rbf[location_mzpredict], c='y', label='RBF')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax3.stem(mz_array[location_mzpredict]+0.5, y_rbf[location_mzpredict], label='SVM rbf')
        else:
            markerline, stemlines, baseline = ax3.stem(mz_array, [0.0]*len(mz_array), label='SVM rbf')
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax3.set_title(numpy.round(total_deviation_rbf_full[step1],decimals=4))
        ax3.legend()
        ax3.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax3.set_xlim(0, 80)

        # Lin
        #ax4.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax4.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax4.plot(mz_array[location_mzpredict], y_lin[location_mzpredict], c='y', label='Lin')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax4.stem(mz_array[location_mzpredict]+0.5, y_lin[location_mzpredict], label='SVM lin')
        else:
            markerline, stemlines, baseline = ax4.stem(mz_array, [0.0]*len(mz_array), label='SVM lin')    
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax4.set_title(numpy.round(total_deviation_lin_full[step1],decimals=4))
        ax4.legend()
        ax4.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax4.set_xlim(0, 80)

        # Poly
        #ax5.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax5.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax5.plot(mz_array[location_mzpredict], y_poly[location_mzpredict], c='y', label='Poly')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax5.stem(mz_array[location_mzpredict]+0.5, y_poly[location_mzpredict], label='SVM poly')
        else:
            markerline, stemlines, baseline = ax5.stem(mz_array, [0.0]*len(mz_array), label='SVM poly')    
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax5.set_title(numpy.round(total_deviation_poly_full[step1],decimals=4))
        ax5.legend()
        ax5.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax5.set_xlim(0, 80)

        # OLS
        #ax6.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax6.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax6.plot(mz_array[location_mzpredict], y_ols[location_mzpredict], c='y', label='OLS')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax6.stem(mz_array[location_mzpredict]+0.5, y_ols[location_mzpredict], label='OLS')
        else:
            markerline, stemlines, baseline = ax6.stem(mz_array, [0.0]*len(mz_array), label='OLS')
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax6.set_title(numpy.round(total_deviation_ols_full[step1],decimals=4))
        ax6.legend()
        ax6.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax6.set_xlim(0, 80)

        # BRR
        #ax7.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax7.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax7.plot(mz_array[location_mzpredict], y_brr[location_mzpredict], c='y', label='Tree')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax7.stem(mz_array[location_mzpredict]+0.5, y_brr[location_mzpredict], label='BRR')
        else:
            markerline, stemlines, baseline = ax7.stem(mz_array, [0.0]*len(mz_array), label='BRR')    
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax7.set_title(numpy.round(total_deviation_brr_full[step1],decimals=4))
        ax7.legend()
        ax7.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax7.set_xlim(0, 80)
    
        # SGDR
        #ax8.scatter(mz_array, EI_matrix_new[numbers_with_names[step1],:], c='k', label='Data')
        markerline, stemlines, baseline = ax8.stem(mz_array, EI_matrix_new[step1,:], '-.', label='Data')
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.hold(True)
        #ax8.plot(mz_array[location_mzpredict], y_sgdr[location_mzpredict], c='y', label='BRR')
        if len(location_mzpredict)>0:
            markerline, stemlines, baseline = ax8.stem(mz_array[location_mzpredict]+0.5, y_sgdr[location_mzpredict], label='SGDR')
        else:
            markerline, stemlines, baseline = ax8.stem(mz_array, [0.0]*len(mz_array), label='SGDR')    
        plt.setp(markerline, 'markerfacecolor', 'r')
        ax8.set_title(numpy.round(total_deviation_sgdr_full[step1],decimals=4))
        ax8.legend()
        ax8.set_ylim(0, max(EI_matrix_new[step1,:]))
        ax8.set_xlim(0, 80)
 
        plt.legend()
        plt.setp(ax1.get_xticklabels(), visible=True)
        plt.setp(ax2.get_xticklabels(), visible=True)
        plt.setp(ax3.get_xticklabels(), visible=True)
        plt.setp(ax4.get_xticklabels(), visible=True)
        plt.setp(ax5.get_xticklabels(), visible=True)
        plt.setp(ax6.get_xticklabels(), visible=True)
        plt.setp(ax7.get_xticklabels(), visible=True)
        plt.setp(ax8.get_xticklabels(), visible=True)
        
        plt.suptitle(s+str(':')+str(step1))
                                                
        plt.show()
        plt.close()
    step1+=1

file_name.close()



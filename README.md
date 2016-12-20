# STRAPS
Supervised Training Regression for the Arbitrary Prediction of Spectra (STRAPS)  

The idea behind this model evaluation facility is as follows

We have a database of EI mass spectra, which gives peak height for any given m/z channel, per compound. Each compound is represented by a SMILES string which is a way of representing a chemical structure For more information on SMILES see:
http://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system

The question we want to answer is: Can we predict EI mass spectra based on the structure of a molecule?? To answer this we take the database of measured spectra and then extract structural features per molecules to see if a range of machine learning algorithms can use these features to fit a model that relates peak height to such features. How we extract these features can vary. We normally use a database of SMARTS strings to interrogate SMILEs strings. These SMARTS strings represent key features. For example, this might include CH2 groups, a COOH group next to a CH2 group, and so on. There are many databases that collate such strings and we use a few in this study. We use our own SMARTS libraries based on models of chemical properties such as vapour pressure as well as more generic libraries defined elsewhere, referred to as MACCS keys or FP4 keys:

http://www.dalkescientific.com/writings/NBN/fingerprints.html

To begin training the models, we want a matrix of identified features per compound as follows:
*SMILES ---[use SMARTS]---> a matrix of identified features ---> input to machine learning algorithms to relate peak height to features ---> a model that can predict peak height when given a SMILES string.*

As you look through this code you will see we will use the UManSysProp facilities to use some of these SMARTS libraries, whilst for more generic SMARTS we extract SMARTS from the RDKit python package. This is simply for efficiency.

The code was developed under Ubuntu and has been tested with the following Python package versions:
sklearn - '0.18'
numpy - '1.11.1'
matplotlib - '1.3.1'
scipy - '0.17.1'

The code also relies on the UManSysProp suite which can be found here: https://github.com/loftytopping/UManSysProp_public

This code focuses on the Aerosol Mass Spectrometer [AMS]. The original spectral library was taken from:
*http://cires1.colorado.edu/jimenez-group/AMSsd/*
where each file now has a SMILES string inserted.

Please see the development paper associated with this code for more details.

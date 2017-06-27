# STRAPS
Supervised Training Regression for the Arbitrary Prediction of Spectra (STRAPS)  

The idea behind this model evaluation facility is as follows

We have a database of EI mass spectra, which gives peak height for any given m/z channel, per compound. Each compound is represented by a SMILES string which is a way of representing a chemical structure For more information on SMILES see:
http://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system

The question we want to answer is: Can we predict EI mass spectra based on the structure of a molecule?? To answer this we take the database of measured spectra and then extract structural features to see if a range of machine learning algorithms can use these features to fit a model that predicts peak height. How we extract these features can vary. We normally use a database of SMARTS strings to interrogate SMILEs strings. These SMARTS strings represent key features. For example, this might include CH2 groups, a COOH group next to a CH2 group, and so on. There are many databases that collate such strings and we use a few in this study. We use our own SMARTS libraries based on models of chemical properties such as vapour pressure as well as more generic libraries defined elsewhere, referred to as MACCS keys or FP4 keys:

http://www.dalkescientific.com/writings/NBN/fingerprints.html

To begin training the models, we want a matrix of identified features per compound as follows:
*SMILES ---[use SMARTS]---> a matrix of identified features ---> input to machine learning algorithms to relate peak height to features ---> a model that can predict peak height when given a SMILES string.*

As you look through this code you will see we will use the UManSysProp facilities to use some of these SMARTS libraries, whilst for more generic SMARTS we extract SMARTS from the RDKit python package. This is simply for efficiency.

The code was developed under Ubuntu and has been tested with the following Python package versions:
sklearn - '0.18'
numpy - '1.11.1'
matplotlib - '1.3.1'
scipy - '0.17.1'

To run the code, simply type: Python Master_Script.py

This code provides you will the statistics on model performance when varying the percentage of data used for training over evaluation. It also provides you will predictions on spectra for compounds not included in the training set.

The code also relies on the UManSysProp suite which can be found here: https://github.com/loftytopping/UManSysProp_public

This code focuses on the Aerosol Mass Spectrometer [AMS]. The original spectral library was taken from:
*http://cires1.colorado.edu/jimenez-group/AMSsd/*
where each file now has a SMILES string inserted. If you use only the .itx files taken from that site, as provided here, you must abide by the citation requirements provided on that site.

If you use this model or component of it, please cite our development paper in Geoscientific Model Development:
STRAPS v1.0: evaluating a methodology for predicting electron impact ionisation mass spectra for the aerosol mass spectrometer
David O. Topping, James Allan, M. Rami Alfarra, and Bernard Aumont
Geosci. Model Dev., 10, 2365-2377, https://doi.org/10.5194/gmd-10-2365-2017, 2017

Code DOI: [![DOI](https://zenodo.org/badge/76975252.svg)](https://zenodo.org/badge/latestdoi/76975252)



##########################################################################################
#											 #
#    Plotting scripts associated with EI spectra predictions                             # 
#                                                                                        #
#                                                                                        #
#    Copyright (C) 2016  David Topping : david.topping@manchester.ac.uk                  #
#                                      : davetopp80@gmail.com                            # 
#    Personal website: davetoppingsci.com                                                #
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
import pdb
import numpy
import matplotlib 
import matplotlib.pyplot as plt


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rc('font', size=16) 

def Visualise_keys(Key_matrix):

    # This is a simple function to visualise the sparsity of features for all
    # compounds used
    
    fig, ax = plt.subplots()
    im = ax.imshow(Key_matrix, cmap=plt.get_cmap('cool'), interpolation='nearest',vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im)
    plt.xlabel('Key number')
    plt.ylabel('Compound number')
    plt.show()
    filename = 'key_coverage.png' 
    plt.savefig(filename)
    plt.close()
    
def Feature_importance_gen(feature_importance):

    # This plots the feature importance as a function of keys versus m/z channels

    fig, ax = plt.subplots()
    im = ax.imshow((feature_importance), cmap=plt.get_cmap('cool'), interpolation='nearest', aspect='auto')
    fig.colorbar(im)
    plt.xlabel('Features')
    plt.ylabel('mz array')
    plt.show()
    filename = 'Feature_coverage.png' 
    plt.savefig(filename)
    plt.close()


def Feature_importance_spec(feature_importance_key_channels,key_channels):

    # This plots the feature importance as a function of hand picked keys versus m/z channels

    fig, ax = plt.subplots()
    im = ax.imshow((feature_importance_key_channels), cmap=plt.get_cmap('cool'), interpolation='nearest', aspect='auto')
    fig.colorbar(im,orientation='horizontal')
    locs, labels = plt.yticks()
    plt.yticks(numpy.arange(len(key_channels)),key_channels)
    plt.xlabel('Features')
    plt.ylabel('mz array')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    #pdb.set_trace()
    filename = 'Feature_coverage_key_channels.png' 
    plt.savefig(filename)
    plt.close() 
    
def Scatter_plot(mz_array,key_location,size_array,total_deviation_tree_matrix,total_deviation_ensemble_tree_matrix,
    total_deviation_brr_matrix,total_deviation_ols_matrix,total_deviation_sgdr_matrix,total_deviation_rbf_matrix,
    total_deviation_poly_matrix,total_deviation_lin_matrix):
    
    # This plots absolute deviation in peak heigh, r squared and dot product for each m/z channel, coloured
    # according to the amount of compounds used in the fitting process.
    
    # First plot all methods for dot product in one figure.
    plt.subplot(3, 3, 1)
    sc1=plt.scatter(mz_array, total_deviation_ensemble_tree_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.ylabel('Dot product')
    plt.title('Ensemble tree')
    plt.ylim((0,1.0))
    plt.subplot(3, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_tree_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.ylabel('Dot product')
    plt.title('Decision tree')
    plt.ylim((0,1.0))    
    plt.subplot(3, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_sgdr_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.ylabel('Dot product')
    plt.title('Stochastic Descent')
    plt.ylim((0,1.0))
    plt.subplot(3, 3, 4)
    sc1=plt.scatter(mz_array, total_deviation_ols_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.ylabel('Dot product')
    plt.title('Ordinary least squares')
    plt.ylim((0,1.0))   
    plt.subplot(3, 3, 5)    
    sc1=plt.scatter(mz_array, total_deviation_brr_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.ylabel('Dot product')
    plt.title('Bayesian Ridge')
    plt.ylim((0,1.0))    
    plt.subplot(3, 3, 6)    
    sc1=plt.scatter(mz_array, total_deviation_rbf_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM rbf')
    plt.ylim((0,1.0))    
    plt.subplot(3, 3, 7)    
    sc1=plt.scatter(mz_array, total_deviation_lin_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM lin')
    plt.ylim((0,1.0))    
    plt.subplot(3, 3, 8)    
    sc1=plt.scatter(mz_array, total_deviation_poly_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM poly')
    plt.ylim((0,1.0))    
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_ensemble_tree_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_ensemble_tree_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_ensemble_tree_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('Ensemble tree')
    plt.ylim((-1.0,1.0))
    plt.show()
    plt.close()
    
    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_tree_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_tree_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_tree_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('Decision tree')
    plt.ylim((-1.0,1.0))    
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_sgdr_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_sgdr_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_sgdr_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('Stochastic Descent')
    plt.ylim((-1.0,1.0))
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_ols_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_ols_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))    
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_ols_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('Ordinary least squares')
    plt.ylim((-1.0,1.0))   
    plt.show()
    filename = 'Ordinary_least_squares_stats_featselect.png' 
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_brr_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_brr_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))    
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_brr_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('Bayesian Ridge')
    plt.ylim((-1.0,1.0))    
    filename = 'Bayesian_Ridge_stats_featselect.png' 
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_rbf_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_rbf_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))    
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_rbf_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM rbf')
    plt.ylim((-1.0,1.0))    
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_lin_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_lin_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))    
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_lin_matrix[:,2],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM lin')
    plt.ylim((-1.0,1.0))    
    plt.show()
    plt.close()

    plt.subplot(1, 3, 1)
    sc=plt.scatter(mz_array, numpy.log10(total_deviation_poly_matrix[:,0]),s=100, c=size_array, label='data')
    plt.colorbar(sc)
    plt.xlabel('m/z ratio')
    plt.ylabel('Average deviation')
    plt.subplot(1, 3, 2)
    sc1=plt.scatter(mz_array, total_deviation_poly_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('R squared')
    plt.ylim((0,1.0))    
    plt.subplot(1, 3, 3)
    sc1=plt.scatter(mz_array, total_deviation_poly_matrix[:,1],s=100, c=size_array, label='data')
    plt.colorbar(sc1)
    plt.xlabel('m/z ratio')
    plt.ylabel('Dot product')
    plt.title('SVM poly')
    plt.ylim((-1.0,1.0))    
    plt.show()
    plt.close()

def Boxplot_all(total_deviation_rbf_full,total_deviation_poly_full,total_deviation_lin_full,total_deviation_brr_full,
    total_deviation_ols_full,total_deviation_sgdr_full,total_deviation_tree_full,total_deviation_ensemble_tree_full):
    
    # This produces a boxplot of dot products over full spectra for all compounds
    
    names = ['Forest','Tree','SVM RBF', 'SVM Lin', 'SVM poly','BRR','OLS','SGDR']
    data=[total_deviation_ensemble_tree_full,total_deviation_tree_full, total_deviation_rbf_full, total_deviation_lin_full, 
    total_deviation_poly_full, total_deviation_brr_full, total_deviation_ols_full, total_deviation_sgdr_full]
    fig, ax1 = plt.subplots(figsize=(10,6))
    bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    xtickNames = plt.setp(ax1, xticklabels=names)
    plt.setp(xtickNames, rotation=0, fontsize=14)
    plt.ylabel('Dot product variability [over all compounds]')
    filename = 'Box_plot_dot_product.png' 
    plt.show()
    plt.close()
    



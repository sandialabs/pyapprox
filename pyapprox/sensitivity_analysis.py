from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from pyapprox.indexing import compute_hyperbolic_indices,hash_array

def get_main_and_total_effect_indices_from_pce(coefficients,indices):
    """
    Assume basis is orthonormal
    Assume first coefficient is the coefficient of the constant basis. Remove
    this assumption by extracting array index of constant term from indices

    Returns
    -------
    main_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone

    total_effects : np.ndarray(num_vars)
        Contribution to variance of each variable acting alone or with 
        other variables

    """
    num_vars = indices.shape[0]
    num_terms,num_qoi = coefficients.shape
    assert num_terms==indices.shape[1]
    
    main_effects = np.zeros((num_vars,num_qoi), np.double )
    total_effects = np.zeros((num_vars,num_qoi), np.double )
    variance = np.zeros(num_qoi)

    for ii in range(num_terms):
        index = indices[:,ii]
        
        # calculate contribution to variance of the index
        var_contribution = coefficients[ii,:]**2
        
        # get number of dimensions involved in interaction, also known
        # as order
        non_constant_vars = np.where(index>0)[0]
        order = non_constant_vars.shape[0]

        if order>0:
            variance += var_contribution
        
        # update main effects
        if ( order == 1 ):
            var = non_constant_vars[0]
            main_effects[var,:] += var_contribution
            
        # update total effects
        for ii in range( order ):
            var = non_constant_vars[ii]
            total_effects[var,:] += var_contribution

    main_effects /= variance
    total_effects /= variance
    return main_effects, total_effects

def get_sobol_indices(coefficients,indices,max_order=2):
    num_terms,num_qoi = coefficients.shape
    variance = np.zeros(num_qoi)
    assert num_terms==indices.shape[1]
    interactions = dict()
    interaction_values = []
    interaction_terms = []
    kk=0 
    for ii in range(num_terms):
        index = indices[:,ii]
        var_contribution = coefficients[ii,:]**2
        non_constant_vars = np.where(index>0)[0]
        key = hash_array(non_constant_vars)

        if len(non_constant_vars) > 0:
            variance += var_contribution
        
        if len(non_constant_vars) > 0 and len(non_constant_vars)<=max_order:
            if key in interactions:
                interaction_values[interactions[key]] += var_contribution
            else:
                interactions[key]=kk
                interaction_values.append(var_contribution)
                interaction_terms.append(non_constant_vars)
                kk+=1

    interaction_terms = np.asarray(interaction_terms).T
    interaction_values = np.asarray(interaction_values)
    
    return interaction_terms, interaction_values/variance

from pyapprox.configure_plots import *
def plot_pie_chart( fracs, labels = None, title = None, show = True,
                    fignum = 1, fontsize = 20, filename = None ):
    plt.figure( fignum )
    plt.pie( fracs, labels = labels, autopct='%1.1f%%',
               shadow = True )#, startangle = 90 )

    environment_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = fontsize

    if ( title is not None ):
        plt.title( title )

    if ( filename is not None ):
        plt.savefig( filename, dpi = 300, format = 'png' )

    if ( show ):
        plt.show()

    mpl.rcParams['font.size'] = environment_fontsize

def plot_main_effects( main_effects, title = None, labels = None, show = True,
                       fignum = 1, fontsize = 20, truncation_pct = 0.95,
                       max_slices = 5, filename = None, rv='z', qoi=0  ):

    main_effects = main_effects[:,qoi]
    assert main_effects.sum()<=1.

    # sort main_effects in descending order
    I = np.argsort( main_effects )[::-1]
    main_effects = main_effects[I]

    labels = []
    partial_sum = 0.
    for i in range( I.shape[0] ):
        if partial_sum < truncation_pct and i < max_slices:
            labels.append( '$%s_{%d}$' %(rv,I[i]+1) )
            partial_sum += main_effects[i]
        else:
            break

    main_effects.resize( i + 1 )
    if abs( partial_sum - main_effects.sum()) > 100*np.finfo(np.double).eps:
        labels.append( 'other' )
        main_effects[-1] =  main_effects.sum() - partial_sum

    plot_pie_chart( main_effects, labels = labels , title = title, show = show,
                    fignum = fignum, fontsize = fontsize, filename = filename )

def plot_total_effects( total_effects, title = None, labels = None,
                        show = True, fignum = 1, fontsize = 20,
                        truncation_pct = 0.95, max_slices = 5,
                        filename = None, rv = 'z', qoi=0  ):

    total_effects = total_effects[:,qoi]

    # normalise total effects
    total_effects /= total_effects.sum()

    # sort total_effects in descending order
    I = np.argsort( total_effects )[::-1]
    total_effects = total_effects[I]

    labels = []
    partial_sum = 0.
    for i in range( I.shape[0] ):
        if partial_sum < truncation_pct*total_effects.sum() and i < max_slices:
            labels.append( '$%s_{%d}$' %(rv,I[i]+1) )
            partial_sum += total_effects[i]
        else:
            break

    total_effects.resize( i + 1 )
    if abs( partial_sum - 1. ) > 10 * np.finfo( np.double ).eps:
        labels.append( 'other' )
        total_effects[-1] = 1. - partial_sum

    plot_pie_chart( total_effects, labels = labels, title = title, show = show,
                    fignum = fignum, fontsize = fontsize, filename = filename )


def plot_interaction_values( interaction_values, interaction_terms,
                             title = None, labels = None,
                             show = True, fignum = 1, fontsize = 20,
                             truncation_pct = 0.95, max_slices = 5,
                             filename = None, rv = 'z', qoi=0 ):

    interaction_values = interaction_values[:,qoi]

    interaction_values /= interaction_values.sum()

    labels = []
    partial_sum = 0.
    for i in range( interaction_values.shape[0] ):
        if partial_sum < truncation_pct and i < max_slices:
            l = '($'
            for j in range( len( interaction_terms[i] )-1 ):
                l += '%s_{%d},' %(rv,interaction_terms[i][j]+1)
            l+= '%s_{%d}$)' %(rv,interaction_terms[i][-1]+1)
            labels.append( l )
            partial_sum += interaction_values[i]
        else:
            break

    interaction_values = interaction_values[:i+1]
    if abs( partial_sum - 1. ) > 10 * np.finfo( np.double ).eps:
        labels.append( 'other' )
        interaction_values[-1] = 1. - partial_sum


    assert interaction_values.shape[0] == len ( labels )
    plot_pie_chart( interaction_values, labels = labels, title = title,
                    show = show, fignum = fignum, fontsize = fontsize,
                    filename = filename )

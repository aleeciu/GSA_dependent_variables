# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:23:27 2019

@author: ciullo
"""
import numpy as np
import pandas as pd
from SALib.sample import sobol_sequence
from scipy.stats import norm, uniform, lognorm, gumbel_r, exponweib

def conditional_sampling(u2, x1, mu_x1, mu_x2, cov, s, lower_cond=True):
    '''
    Find vector x2 conditioned on vector x1
     
    --------
    u2: 
        independent random vector in [0,1] used to create the conditional variables
         
    x1: 
        independent variables
         
    mu_x1, mu_x2, cov: 
        mean of independent variables; mean of conditional variables; covariance matrix
         
    s:
        splitting position between independent and conditional varaibles
         
    lower_cond: 
         if the the conditional variables are those after position s. Default is True.
    '''
                  
    # if x2 is the lower vector:
    if lower_cond:
        x1i, x1f = 0, s
        x2i, x2f = s, None
        
    else:
        x1i, x1f = s, None
        x2i, x2f = 0, s

    covx1 = cov[x1i:x1f, x1i:x1f]
    covx2x1 = cov[x2i:x2f, x1i:x1f]
     
    # conditional mu and covariance:           
    mu_x2_c = mu_x2 + covx2x1.dot(np.linalg.inv(covx1)).dot(x1.T-mu_x1)
    cov_x2_c = np.linalg.inv(np.linalg.inv(cov)[x2i:x2f, x2i:x2f])

    L_x2_c = np.linalg.cholesky(cov_x2_c)
    x2_c = mu_x2_c + np.dot(L_x2_c, norm.ppf(u2).T)
     
    return x2_c.T


def kucherenko_sampling(problem, N, cov, mu, s=1):
    '''
    Implementation of the alghoritm proposed in:
     
    S. Kucherenko, S. Tarantola, P. Annoni. Estimation of global sensitivity indices for models with dependent variables
    Comput. Phys. Commun., 183 (4) (2012), pp. 937-946
     
    to generate two sets of independent and conditional variables.
     
    --------
    problem: dict
         
    N: int
         independent variables
         
    mu, cov: 
         mean of variables; covariance matrix
         
    s: int
        splitting position between independent and conditional varaibles
    '''    
    
    factors_order = cov.columns.values
    D = problem['num_vars']
    
    # How many values of the Sobol sequence to skip
    skip_values = 1000

    base_sequence = sobol_sequence.sample(N+skip_values, 2 * D)
    u = base_sequence[skip_values:, :D]
    u_ = base_sequence[skip_values:, D:]
    
    zu = norm.ppf(u)
    L = np.linalg.cholesky(cov)
    x = mu + np.dot(L, zu.T)
    
    v_ = u_[:, :s]
    w_ = u_[:, s:]
    y = x.T[:, :s]
    z = x.T[:, s:]
        
    cov_new = np.cov(x)
    
    mu_y = np.mean(y, axis = 0).reshape(y.shape[1], 1)
    mu_z = np.mean(z, axis = 0).reshape(z.shape[1], 1)

    zc_n = conditional_sampling(w_, y, mu_y, mu_z, cov_new, s, True)
    yc_n = conditional_sampling(v_, z, mu_z, mu_y, cov_new, s, False)
        
    x_df = pd.DataFrame(np.hstack([y, z]), columns = factors_order)
    xc_df = pd.DataFrame(np.hstack([yc_n, zc_n]), columns = factors_order)
                
    return x_df, xc_df

def sobol_indexes(fun, x, xc, problem, s=1):
    '''
    Compute Sobol Indexes
     
    --------
    fun: function
     
    x: array_like
        The independent vector
        
    xc: array_like
         The conditional vector
         
    problem: dict
         
    s: int
        splitting position between independent and conditional variables
    '''        
        
    fnc = x.columns[:s]
    fc = x.columns[s:]
        
    x = x[np.sort(x.columns)]
    xc = xc[np.sort(x.columns)]
    
    # get the marginals' distributions
    for dist in np.unique(problem['dist']):
        logical = problem['dist'] == dist
        
        cols = x.columns[logical]
        prms = problem['prms'][logical]
                
        x[cols] = to_marginal(x[cols], dist, prms)
        xc[cols] = to_marginal(xc[cols], dist, prms)
        
    y_zc = pd.concat([x[fnc], xc[fc]], axis=1)[np.sort(x.columns)]
    yc_z = pd.concat([xc[fnc], x[fc]], axis=1)[np.sort(x.columns)]

    f_y_z = fun(x.values)

    f_y_zc = fun(y_zc.values)
    f_yc_z = fun(yc_z.values)
    
    mean_sq_yz = np.mean(f_y_z**2)
    sq_mean_yz = np.mean(f_y_z)**2
    Vy = mean_sq_yz - sq_mean_yz
    
    # Equation 5.3
    Sy = (np.mean(f_y_z*f_y_zc) - sq_mean_yz)/Vy
    
    # Equation 5.4
    STy = np.mean((f_y_z - f_yc_z)**2)/(2*Vy)

    return Sy, STy

def to_marginal(x, dist, prms):
    dist = eval(dist)
    
    if dist == norm:
        pass
    
    elif dist == uniform:
        loc = prms[:,0]
        scale = prms[:,1] - loc
                
        x = pd.DataFrame(dist.ppf(norm.cdf(x), loc=loc, scale=scale), 
                        columns=x.columns)

    elif dist == gumbel_r:
        loc = prms[:,0]
        scale = prms[:,1]
        
        x = pd.DataFrame(dist.ppf(norm.cdf(x), loc=loc, scale=scale), 
                        columns=x.columns)

    elif dist == exponweib:
        loc = prms[:,0]
        scale = prms[:,1]
        shp1 = prms[:,2]
        shp2 = prms[:,3]
        
        x = pd.DataFrame(dist.ppf(norm.cdf(x), shp1, shp2, loc=loc, scale=scale), 
                        columns=x.columns)        
        
    elif dist == lognorm:
        loc = prms[:,0]
        scale = prms[:,1]
        shp1 = prms[:,2]
        
        x = pd.DataFrame(dist.ppf(norm.cdf(x), shp1, loc=loc, scale=scale), 
                        columns=x.columns)
    else:
        raise('Unknown distribution')
    
    return x

def build_cov_mu(cov, mu, factors):
    '''
    Return ordered covariance matrix
    
    ------
    cov: DataFrame
         covariance matrix in the original order
         
    mu: array_like
         vector of mean values in the original order         
         
    factors: list
         list of input variables of interest   
    '''
    
    new_factors_order = factors + [f for f in cov.columns if not f in factors]
    ordered_cov = cov.loc[new_factors_order, new_factors_order]
    ordered_mu = mu[new_factors_order]
    
    return ordered_cov, ordered_mu

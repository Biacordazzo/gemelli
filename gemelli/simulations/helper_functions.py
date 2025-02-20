import biom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from gemelli.simulations import (Homoscedastic,
                                 Heteroscedastic,
                                 Subsample)

def block_diagonal_gaus(
        ncols,
        nrows,
        nblocks,
        overlap=0,
        minval=0,
        maxval=1.0,
        last_sigma=4):
    """
    Generate block diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    ncol : int
        Number of columns

    nrows : int
        Number of rows

    nblocks : int
        Number of blocks, mucst be greater than one

    overlap : int
        The Number of overlapping columns (Default = 0)

    minval : int
        The min value output of the table (Default = 0)

    maxval : int
        The max value output of the table (Default = 1)


    Returns
    -------
    np.array
        Table with a block diagonal where the rows represent samples
        and the columns represent features.  The values within the blocks
        are gaussian distributed between 0 and 1.
    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    if nblocks <= 1:
        raise ValueError('`nblocks` needs to be greater than 1.')
    mat = np.zeros((nrows, ncols))
    gradient = np.linspace(0, 10, nrows)
    mu = np.linspace(0, 10, ncols)
    sigma = 1
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    mat = np.vstack(xs).T

    block_cols = ncols // nblocks
    block_rows = nrows // nblocks
    for b in range(nblocks - 1):

        gradient = np.linspace(
            5, 5, block_rows)  # samples (bock_rows)
        # features (block_cols+overlap)
        mu = np.linspace(0, 10, block_cols + overlap)
        sigma = 2.0
        xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
              for i in range(len(mu))]

        B = np.vstack(xs).T * maxval
        lower_row = block_rows * b
        upper_row = min(block_rows * (b + 1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols * (b + 1), ncols)

        if b == 0:
            mat[lower_row:upper_row,
                lower_col:int(upper_col + overlap)] = B
        else:
            ov_tmp = int(overlap / 2)
            if (B.shape) == (mat[lower_row:upper_row,
                                 int(lower_col - ov_tmp):
                                 int(upper_col + ov_tmp + 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp + 1)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp - 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp - 1)] = B

    upper_col = int(upper_col - overlap)
    # Make last block fill in the remainder
    gradient = np.linspace(5, 5, nrows - upper_row)
    mu = np.linspace(0, 10, ncols - upper_col)
    sigma = last_sigma
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    B = np.vstack(xs).T * maxval

    mat[upper_row:, upper_col:] = B

    return mat

def shape_noise(X, 
                fxs,
                f_intervals,
                s_intervals,
                n_timepoints=10,
                col_handle='individual'):
    """
    Adds x-shaped noise (e.g. sine, sigmoid) to the
    true data

    Parameters
    ----------
    X : np.array
        The true data

    fxs : list
        List of functions to apply to the data

    f_intervals : list
        List of tuples of the form (f1, f2) where
        f1 is the start index and f2 is the end index
        of the features to apply the function to
    
    s_intervals : list
        List of tuples of the form (s1, s2) where
        s1 is the start index and s2 is the end index 
        of the samples to apply the function to
    
    n_timepoints : int
        Number of timepoints per individual
        Assumes that all individuals have the 
        same number of timepoints
    
    col_handle : str
        How to handle  (individuals)
        'individual': apply function to all 
            timepoints in each individual
        'all': apply function to all columns
    
    Returns
    -------
    np.array
        The data with x-shaped noise added
    """

    #get shape of true data
    rows, cols = X.shape

    #loop through functions
    for func, features, individuals in \
        zip(fxs, f_intervals, s_intervals):

        for f_coord, s_coord in zip(features, individuals):
            f1, f2 = f_coord
            s1, s2 = tuple(int(idx/n_timepoints) for idx in s_coord)
            #get sample subset
            if col_handle == 'individual':
                #loop through individuals
                for i in range(s1, s2):
                    idx1 = i*n_timepoints
                    idx2 = (i+1)*n_timepoints
                    X_sub = X[f1:f2, idx1:idx2]
                    X_sub_noise = np.apply_along_axis(func, 
                                                      tps=10, 
                                                      axis=1, 
                                                      arr=X_sub)
                    #update data
                    X[f1:f2, idx1:idx2] = X_sub_noise
            else:
                X_sub = X[f1:f2, :]
                X_sub_noise = np.apply_along_axis(func, 
                                                  tps=cols, 
                                                  axis=1, 
                                                  arr=X_sub)
                #update data
                X[f1:f2, :] = X_sub_noise
    return X

def build_block_model(
        rank,
        hoced,
        hsced,
        spar,
        C_,
        num_samples,
        num_features,
        num_timepoints,
        col_handle='individual',
        overlap=0, last_sigma=4,
        fxs=None,
        f_intervals=None,
        s_intervals=None,
        mapping_on=True,
        X_noise=None):
    """
    Generates hetero and homo scedastic noise on base truth block
    diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    rank : int
        Number of blocks

    hoced : int
        Amount of homoscedastic noise

    hsced : int
        Amount of heteroscedastic noise

    inten : int
        Intensity of the noise

    spar : int
        Level of sparsity

    C_ : int
        Intensity of real values

    num_features : int
        Number of rows

    num_samples : int
        Number of columns

    num_timepoints : int
        Number of timepoints per individual. Assumes all
        individuals have the same number.

    col_handle : str
        How to handle  (individuals)
        'individual': apply function to all 
            timepoints in each individual
        'all': apply function to all columns

    overlap : int
        The Number of overlapping columns (Default = 0)

    fxs : list
        List of functions to apply to the data

    f_intervals : list
        List of tuples of the form (f1, f2) where
        f1 is the start index and f2 is the end index
        of the features to apply the function to
    
    s_intervals : list
        List of tuples of the form (s1, s2) where
        s1 is the start index and s2 is the end index 
        of the samples to apply the function to

    mapping_on : bool
        if true will return pandas dataframe mock mapping file by block

    X_noise: np.array, default is None
        Data with pre-added gaussian noise. Use this to ensure
        the same underlying data is used for multiple simulations

    Returns
    -------
    Pandas Dataframes
    Table with a block diagonal where the rows represent samples
    and the columns represent features.  The values within the blocks
    are gaussian.

    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    # make a mock OTU table
    X_true = block_diagonal_gaus(num_samples,
                                 num_features,
                                 rank, overlap,
                                 minval=.01,
                                 maxval=C_,
                                 last_sigma=last_sigma)
    if X_noise is None:
        if mapping_on:
            # make a mock mapping data
            mappning_ = pd.DataFrame(np.array([['Cluster %s' % str(x)] *
                                            int(num_samples / rank)
                                            for x in range(1,
                                            rank + 1)]).flatten(),
                                    columns=['example'],
                                    index=['sample_' + str(x)
                                            for x in range(1, num_samples+1)])
        X_noise = X_true.copy()
        X_noise = np.array(X_noise)
        # add Homoscedastic noise
        X_noise = Homoscedastic(X_noise, hoced)
        # add Heteroscedastic noise
        X_noise = Heteroscedastic(X_noise, hsced)
        # Induce low-density into the matrix
        X_noise = Subsample(X_noise, spar, num_samples)
    
    if fxs is not None:
        X_signal = X_noise.copy()
        # introduce specific signal(s)
        X_signal = shape_noise(X_signal, fxs,
                               f_intervals, s_intervals,
                               n_timepoints=num_timepoints,
                               col_handle=col_handle)
    else:
        X_signal = X_noise.copy()
    
    # return the base truth and noisy data
    if mapping_on:
        return X_true, X_noise, X_signal, mappning_
    else:
        return X_true, X_noise, X_signal
    
def create_sim_data(feature_prefix, n_timepoints, 
                    n_subjects, fxs=None, 
                    f_intervals=None, 
                    s_intervals=None,
                    col_handle='individual', 
                    rank=3, hoced=20, hsced=20, 
                    spar=2e3, C_=2e3, 
                    overlap=0, last_sigma=4,
                    num_samples=48, num_features=500,
                    X_noise=None, mapping_on=False,
                    plotting=False):
    
    #create simulated data
    total_samples = n_timepoints * n_subjects
    (X_true, 
     X_noise, 
     X_signal) = build_block_model(rank=rank, 
                                    hoced=hoced, hsced=hsced, 
                                    spar=spar, C_=C_, 
                                    num_samples=num_samples,
                                    num_features=num_features,
                                    num_timepoints=n_timepoints,
                                    col_handle=col_handle,
                                    overlap=overlap, 
                                    last_sigma=last_sigma, 
                                    fxs=fxs,
                                    f_intervals=f_intervals, 
                                    s_intervals=s_intervals, 
                                    X_noise=X_noise,
                                    mapping_on=mapping_on)
    #add feature and sample IDs
    feat_ids = [ '%sF%d' % (feature_prefix, i+1)
                for i in range(X_signal.shape[0])]
    samp_ids = ['sample%d' % (i+1) for i in range(X_signal.shape[1])]
    X_signal = biom.Table(X_signal, feat_ids, samp_ids)
    X_true = biom.Table(X_true, feat_ids, samp_ids)
    #create metadata
    mf = pd.DataFrame({'timepoint': np.tile(np.arange(n_timepoints), n_subjects)},
                      index=samp_ids)
    ind_ids = [['ind{}'.format(i+1)] * n_timepoints for i in range(n_subjects)]
    ind_ids = np.concatenate(ind_ids)
    mf['ind_id'] = ind_ids
    #add group ids based on rank
    group_ids = []
    for i in range(rank):
        group_ids.extend(['group{}'.format(i+1)] * (total_samples//rank))
    mf['group'] = group_ids
    
    if plotting:
        # visualize simulated data
        # first, the noiseless data
        fig, axn = plt.subplots(1, 3, figsize=(15, 4))
        sns.heatmap(X_true.to_dataframe(), robust=True, ax=axn[0],
                    xticklabels=False, yticklabels=False)
        axn[0].set_title('True data', color='black', fontsize=14)
        # second, the true data after introducing noise
        sns.heatmap(X_noise, robust=True, ax=axn[1], 
                    xticklabels=False, yticklabels=False)
        axn[1].set_title('Post-noise', color='black', fontsize=14)
        # third, noisy data plus introduced signal(s)
        sns.heatmap(X_signal.to_dataframe(), robust=True, ax=axn[2],
                    xticklabels=False, yticklabels=False)
        axn[2].set_title('Post-signal', color='black', fontsize=14)
        # set common labels
        plt.setp(axn, xlabel="samples", ylabel="features")
        plt.show()

    return X_true, X_noise, X_signal, mf

def plot_mean_signal(table, 
                     features=None, 
                     samples=None):
    
    n_features = table.to_dataframe().shape[0]
    n_samples = table.to_dataframe().shape[1]
    
    if features is None and samples is None:
        feat_min = 0
        feat_max = n_features//2
        features_df = table.to_dataframe().values.T
        x = range(0, n_samples, 1)

    elif features is None and samples is not None:
        feat_min = 0
        feat_max = n_features//2
        samp_min, samp_max = samples
        features_df = table.to_dataframe().values[:, samp_min:samp_max].T
        x = range(samp_min, samp_max, 1)
    
    elif features is not None and samples is None:
        feat_min, feat_max = features
        features_df = table.to_dataframe().values[feat_min:feat_max, :].T
        x = range(0, n_samples, 1)

    else:
        feat_min, feat_max = features
        samp_min, samp_max = samples
        features_df = table.to_dataframe().values[feat_min:feat_max, 
                                                  samp_min:samp_max].T
        x = range(samp_min, samp_max, 1)

    # Calculate mean signal and standard deviation
    mean_signal = np.mean(features_df, axis=1)
    std_deviation = np.std(features_df, axis=1)

    # Plotting
    #fig = plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()
    ax.plot(x, mean_signal, label='Mean Signal', color='black')
    # Plot shaded areas for individual features
    for i in range(250):
        ax.fill_between(x, mean_signal - std_deviation, 
                            mean_signal + std_deviation, 
                            alpha=0.2, color='lightsteelblue')
    ax.set_xlabel('Sample #')
    ax.set_ylabel('Expression')
    ax.set_title('Mean feature signal ({}:{})'.format(feat_min, feat_max))
    plt.show()

def plot_loadings(loadings, mf, group_colors, 
                  title, comp1, comp2, mod_name, 
                  ft_groups, ft_names):
    
    if 'component' in comp1:
        comp_lst = ['component_1', 'component_2', 'component_3']
    elif 'PC' in comp1:
        comp_lst = ['PC1', 'PC2', 'PC3']

    state_loadings, feat_loadings, ind_loadings = loadings
    mf = mf.groupby('ind_id').agg({'group':'first'})
    #get feature loading groups
    ft_col = []
    for indexes, group in zip(ft_groups, ft_names):
        index_range = range(indexes[0],indexes[1])
        ft_col.extend([group] * len(index_range))
    #add feature group to feature loadings
    feat_loadings[mod_name]['group'] = ft_col
    
    fig, axn = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
    axn[0].plot(state_loadings[mod_name][comp_lst])
                #label=['PC1', 'PC2', 'PC3'])
    axn[0].set_title('Temporal Loadings', fontsize=14)
    axn[0].legend(['PC1', 'PC2', 'PC3'])
    axn[0].set_xlabel('Timepoint')
    axn[0].set_ylabel('Loadings')
    #plot subject loadings
    axn[1].scatter(ind_loadings[mod_name][comp1],
                   ind_loadings[mod_name][comp2],
                   c=mf['group'].map(group_colors))
    axn[1].set_title('Subject Loadings', fontsize=14)
    axn[1].set_xlabel(comp1.replace('_', ' '))
    axn[1].set_ylabel(comp2.replace('_', ' '))
    #plot feature loadings
    axn[2].scatter(feat_loadings[mod_name][comp1],
                   feat_loadings[mod_name][comp2],
                   edgecolors=feat_loadings[mod_name]['group'].map(group_colors),
                   facecolors='none', alpha=0.5)
    axn[2].set_title('Feature Loadings', fontsize=14)
    axn[2].set_xlabel(comp1.replace('_', ' '))
    axn[2].set_ylabel(comp2.replace('_', ' '))
    plt.suptitle('{} ({})'.format(title, mod_name), 
                 fontsize=16, y=1.02)
    plt.show()

def plot_loadings_v2(loadings, mf, group_colors, 
                     comp1, comp2, mod_name):
    
    if 'component' in comp1:
        comp_lst = ['component_1', 'component_2', 'component_3']
    elif 'PC' in comp1:
        comp_lst = ['PC1', 'PC2', 'PC3']

    state_loadings, ind_loadings = loadings
    mf = mf.groupby('ind_id').agg({'group':'first'})
    
    fig, axn = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    axn[0].plot(state_loadings[mod_name][comp_lst])
                #label=['PC1', 'PC2', 'PC3'])
    axn[0].set_title('Temporal Loadings', fontsize=14)
    axn[0].legend(['PC1', 'PC2', 'PC3'])
    axn[1].scatter(ind_loadings[mod_name][comp1],
                   ind_loadings[mod_name][comp2],
                   c=mf['group'].map(group_colors))
    axn[1].set_title('Subject Loadings', fontsize=14)
    plt.suptitle('Joint-CTF results ({})'.format(mod_name), 
                 fontsize=16, y=1.02)
    plt.setp(axn, xlabel=comp1.replace('_', ' '), 
                  ylabel=comp2.replace('_', ' '))
    plt.show()

def plot_loadings_det(loadings, resolution, mod_name):
    
    x = range(0, resolution+1)
    fig, axn = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
    axn[0].plot(x, loadings[mod_name][['component_1']], 
                color='tab:blue')
    axn[0].set_title("PC1")
    axn[1].plot(x, loadings[mod_name][['component_2']], 
                color='tab:orange')
    axn[1].set_title("PC2")
    axn[2].plot(x, loadings[mod_name][['component_3']], 
                color='tab:green')
    axn[2].set_title("PC3")
    plt.suptitle('Joint-CTF temporal loadings ({})'.format(mod_name), 
                 fontsize=16, y=1.02)
    plt.setp(axn, xlabel='resolution', ylabel='loadings')
    plt.show()

def plot_feature_cov(tables, intervals, 
                     mod_cov, components,
                     sharey=True):
    
    #get feature IDs for each modality
    feature_order = []
    for table, interval in zip(tables, intervals):
        feature_ids = table.ids(axis='observation').tolist()
        fmin, fmax = interval
        feature_order = feature_order + feature_ids[fmin:fmax]

    n_comp = len(components)
    fig, axn = plt.subplots(1, n_comp, figsize=(5*n_comp, 4), 
                            sharey=sharey)
    
    for i, component in enumerate(components):
        cov_table = mod_cov[component]
        cov_table = cov_table.loc[feature_order, feature_order]
        #plot heatmap of feature covariance matrix
        sns.heatmap(cov_table, robust=True, cmap='vlag', 
                    center=0, ax=axn[i])
        axn[i].set_title('{}'.format(component))
    plt.suptitle('Feature covariance', fontsize=16, y=1.02)
    plt.show()

def plot_feature_cov2(tables, mod_cov, components,
                      sharey=True, axis_off=True):
    
    #get feature IDs for each modality
    feature_order = []
    for table in tables:
        feature_ids = table.ids(axis='observation').tolist()
        feature_order = feature_order + feature_ids

    n_comp = len(components)
    fig, axn = plt.subplots(1, n_comp, figsize=(5*n_comp, 4), 
                            sharey=sharey)

    for i, component in enumerate(components):
        cov_table = mod_cov[component]
        cov_table = cov_table.loc[feature_order, feature_order]
        #plot heatmap of feature covariance matrix
        sns.heatmap(cov_table, robust=True, cmap='vlag', 
                    center=0, ax=axn[i])
        axn[i].set_title('{}'.format(component))
    
    if axis_off:
        for ax in axn:
            ax.set_yticks([])
            ax.set_xticks([])

    plt.suptitle('Feature covariance', fontsize=16, y=1.02)
    #plt.show()
    
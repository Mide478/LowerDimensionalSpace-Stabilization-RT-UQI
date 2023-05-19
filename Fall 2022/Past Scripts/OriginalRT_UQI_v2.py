import math
import random
import pandas as pd
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from pydist2.distance import pdist1
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.stats import norm
from shapely.geometry import Polygon
from sklearn.manifold import MDS  # multidimensional scaling
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# TURN OFF ALL GRIDS either via sns or plt.
# noinspection PyTypeChecker
sns.set_style("whitegrid", {'axes.grid': False})


# noinspection PyStatementEffect
def matrix_scatter(dataframe, feat_title, left_adj, bottom_adj, right_adj, top_adj, wspace, hspace, title, palette_,
                   hue_=None, n_case=True, save=True):
    """
    This function plots the matrix scatter plot for the given data.

    Arguments
    ---------

    dataframe: dataframe

    feat_title: a list consisting of a string column names for each predictor feature of choice

    left_adj: float values that adjusts the left placement of the scatter plot

    bottom_adj: float values that adjusts the bottom placement of the scatter plot

    right_adj: float values that adjusts the right placement of the scatter plot

    top_adj: float values that adjusts the top placement of the scatter plot

    wspace: float values that adjusts the width placement of the scatter plot

    hspace: float values that adjusts the height placement of the scatter plot

    title: a string consisting of the name of the figure

    palette_: an integer that assigns a dictionary of colors that maps the hue variable consisting of the
    classification label

    hue_: string variable that is used to color matrix scatter plot made

    n_case:

    save:
    """

    # Hue assignment
    if hue_ is not None:
        hue_ = hue_

        # Palette assignment
        if palette_ == 1:
            palette_ = sns.color_palette("rocket_r", n_colors=len(dataframe[hue_].unique()) + 1)

        elif palette_ == 2:
            palette_ = sns.color_palette("bright", n_colors=len(dataframe[hue_].unique()) + 1)
        else:
            palette_ = None

    else:
        hue_ is None

        # Palette assignment
        if palette_ == 1:
            palette_ = sns.color_palette("rocket_r")

        elif palette_ == 2:
            palette_ = sns.color_palette("bright")
        else:
            palette_ = None

    # For N_case visuals
    if n_case is True:
        sns.pairplot(dataframe, vars=feat_title, markers='o', diag_kws={'edgecolor': 'black'},
                     plot_kws=dict(s=50, edgecolor="black", linewidth=0.5), hue=hue_, corner=True,
                     palette=palette_)

    else:
        # Define marker type for last datapoint i.e., the additional sample in N+1 case
        last_marker = '*'

        # Create pairplot
        fig = sns.pairplot(data=dataframe, vars=feat_title, diag_kws={'edgecolor': 'black'},
                           plot_kws=dict(s=50, edgecolor="black", linewidth=0.5), hue=hue_, corner=True,
                           markers='o', palette=palette_)

        # Plot the last datapoint with a different marker type in all subplots
        for i in range(len(feat_title)):
            for j in range(len(feat_title)):
                if i == j:
                    continue
                ax = fig.axes[i, j]
                if ax is not None:
                    last_datapoint = dataframe[feat_title].iloc[-1, [j, i]].values

                    if dataframe[hue_][len(dataframe) - 1] == 'low':
                        ax.scatter(last_datapoint[0], last_datapoint[1], marker=last_marker, s=200, color=palette_[0],
                                   edgecolors="black", linewidth=0.5)

                    elif dataframe[hue_][len(dataframe) - 1] == 'med':
                        ax.scatter(last_datapoint[0], last_datapoint[1], marker=last_marker, s=200, color=palette_[1],
                                   edgecolors="black", linewidth=0.5)

                    elif dataframe[hue_][len(dataframe) - 1] == 'high':
                        ax.scatter(last_datapoint[0], last_datapoint[1], marker=last_marker, s=200, color=palette_[2],
                                   edgecolors="black", linewidth=0.5)

                    elif dataframe[hue_][len(dataframe) - 1] == 'vhigh':
                        ax.scatter(last_datapoint[0], last_datapoint[1], marker=last_marker, s=200, color=palette_[3],
                                   edgecolors="black", linewidth=0.5)

    plt.subplots_adjust(left=left_adj, bottom=bottom_adj, right=right_adj, top=top_adj, wspace=wspace, hspace=hspace)
    if save:
        plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


def make_levels(data, cat_response, num_response):

    bins = [0, 2500, 5000, 7500, 10000]                              # assign the production bins (these are the fence posts)
    labels = ['low', 'med', 'high', 'vhigh']                     # assign the labels
    category = pd.cut(data[num_response], bins, labels=labels)     # make the 1D array with the labels for our data
    data[cat_response] = category                                # add the new ordinal production feature to our DataFrames

    return data


# noinspection PyTypeChecker
def standardizer(dataset, features, keep_only_std_features=False):
    """
    This function standardizes  the dataframe of choice to a mean of 0 and variance of 1 whilst preserving its natural
    distribution shape.

    Arguments
    ---------
    dataset: DataFrame
        A pandas.DataFrame containing the features of interest
    features: list
        A list consisting of features column names to be normalized
    keep_only_std_features: bool
        True to discard non-normalized features.
    """

    is_string = isinstance(features, str)
    if is_string:
        features = [features]

    df = dataset.copy()
    x = df.loc[:, features].values
    xs = StandardScaler().fit_transform(x)

    ns_feats = []
    for i, feature in enumerate(features):
        df['NS_' + feature] = xs[:, i]
        ns_feats.append('NS_' + feature)

    if keep_only_std_features:
        df = df.loc[:, ns_feats]

    return df


# noinspection PyTypeChecker
def normalizer(array):
    arr = array.copy()
    df = pd.DataFrame(arr)
    feats = df.columns.tolist()
    x = df.loc[:, feats].values
    scaler = MinMaxScaler(feature_range=(-4, 4))
    xs = scaler.fit_transform(x)

    ns_feats = []
    for i in range(0, len(feats)):
        df[arr.shape[1] + feats[i]] = xs[:, i]
        ns_feats.append(arr.shape[1] + feats[i])

    final_array = df.iloc[:, len(feats):].values
    return final_array


def generate_random_seeds(seed, num_realizations, lower_bound, upper_bound):
    """
    :param seed: TODO
    :param num_realizations: TODO
    :param lower_bound: TODO
    :param upper_bound: TODO
    :return: TODO
    """
    random_seeds = []
    random.seed(seed)  # random number seed is set to ensure reproducibility of same realization seeds,
    # which is different for every iteration in the realization
    for i in range(num_realizations):
        random_value = random.randint(lower_bound, upper_bound)
        random_seeds.append(random_value)
    return random_seeds


def rigid_transform_2D(A, B, verbose=False):
    """
    Performs a rigid transformation (rotation and translation) on 2D point sets A and B.

    :param A: 2xN matrix of points
    :param B: 2xN matrix of points
    :return: R: 2x2 rotation matrix, t: 2x1 translation vector

    Parameters
    ----------

    verbose
    """

    assert A.shape == B.shape

    num_rows, num_cols = A.shape

    if num_rows != 2:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # Find the centroids (mean) of each point set
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Center the point sets by subtracting the centroids
    centered_A = A - np.expand_dims(centroid_A, axis=1)
    centered_B = B - np.expand_dims(centroid_B, axis=1)

    # Perform SVD on the centered point sets
    H = centered_A @ centered_B.T
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix R
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0 and verbose:
        print("det(R) < 0, reflection detected!, solution corrected for it")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Calculate the translation vector t
    t = centroid_B - R @ centroid_A

    return R, t


def rigid_transform_3D(A, B, verbose=False):
    """
    This function fits a rigid transform to a set of 3D points.

    # Returns:
    # R: 3×3 rotation matrix
    # t: 3×1 translation column vector

    :param A: 3xN matrices of points
    :param B: 3xN matrices of points
    :param verbose: TODO
    :return: TODO
    """

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    #     print(centroid_A.shape, A.shape)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    #     print(centroid_A.shape, A.shape)
    centroid_B = centroid_B.reshape(-1, 1)

    # centre the points by subtracting the mean to remove translation components
    Am = A - centroid_A
    #     print(Am.shape)
    Bm = B - centroid_B

    # dot is matrix multiplication for array that finds the rotation from A to B
    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    #     print("Vt",Vt)
    #     print("U",U)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0 and verbose:
        print("det(R) < 0, reflection detected!, solution corrected for it")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ (centroid_A + centroid_B)

    #     print("Rotation",R)
    #     print("translation",t)
    #     print("centroidA", centroid_A.shape)
    #     print("centroidB", centroid_B.shape)
    return R, t


# noinspection PyUnboundLocalVariable
def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D points is 'strictly convex': points are valid,
    side lengths non-zero, interior angles are strictly between zero and a straight angle, and the polygon does not
    intersect itself. Checks if data is a convex polygon

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """

    TWO_PI = 2 * math.pi
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -math.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > math.pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return True
        # return False # any exception means not a proper convex polygon


def rmse(array1, array2):
    """

    :param array1: recovered realization "i" array from R, T calculation
    :param array2: base case
    :return: TODO
    """

    var1 = np.transpose(array1) - array2
    var1 = var1 * var1
    var1 = np.sum(var1)
    rmse_error = np.sqrt(var1 / len(array1[0, :]))
    return rmse_error


def make_sample_within_ci(dataframe):
    """
    Sample a single row from a dataframe of multiple columns such that it is within a 95% confidence interval (CI).
    Args:
        dataframe: pandas DataFrame
    Returns:
        A pandas DataFrame with a single row of sampled values from each column, such that each value falls within
        a 95% confidence interval of the original dataset.
    """

    # Set random seed for reproducibility
    random_seed = np.random.randint(0, 100000)
    np.random.seed(random_seed)

    # Calculate mean, standard deviation, and bounds for each column
    n = len(dataframe)
    means = dataframe.mean()
    stds = dataframe.std()
    t = 1.96  # 95% confidence interval for a normal distribution
    lower_bounds, upper_bounds = means - t * (stds / np.sqrt(n)), means + t * (stds / np.sqrt(n))

    # Generate random values within 95% CI for each column
    samples = np.random.uniform(lower_bounds, upper_bounds)

    # Combine sampled values into single row
    sampled_row = pd.DataFrame([samples], columns=dataframe.columns)

    # Add to dataframe
    data = dataframe.copy().append(sampled_row, ignore_index=True)
    return data, random_seed


# noinspection PyUnboundLocalVariable
class RigidTransformation:
    def __init__(self, df, features, idx, num_realizations, base_seed, start_seed, stop_seed, dissimilarity_metric,
                 dim_projection):
        """

        Parameters: TODO
        ----------
        df
        features
        idx: str
        num_realizations
        base_seed
        start_seed
        stop_seed
        dissimilarity_metric
        dim_projection # Based on user input of the LDS i.e., if 3d or 2d
        """
        self.df_idx = df.copy()
        self.df = standardizer(df, features, keep_only_std_features=True)
        self.df_idx[idx] = np.arange(1, len(self.df) + 1).astype(int)
        self.idx = idx
        self.num_realizations = num_realizations
        self.base_seed = base_seed
        self.start_seed = start_seed
        self.stop_seed = stop_seed
        self.dissimilarity_metric = dissimilarity_metric
        self.dim_projection = dim_projection.upper()

        self.random_seeds = None
        self.all_real = None
        self.calc_real = None
        self.all_rmse = None
        self.norm_stress = None
        self.array_exp = None


    def run_rigid_MDS(self, normalize_projections=True):
        """

        Parameters:  TODO
        ----------
        num_realizations
        Returns
        -------
        """
        # Arrays below store random values for every parameter changing using the utility functions defined later in the
        # code for each realization
        random_seeds = generate_random_seeds(self.base_seed, self.num_realizations, self.start_seed, self.stop_seed)

        mds1 = []  # MDS projection 1
        mds2 = []  # MDS projection 2
        norm_stress = []
        all_real = []  # All realizations prepared for rigid transform
        t = []
        r = []
        all_rmse = []
        calc_real = []  # analytical estimation of each realization from R,T recovered

        # Based on user-input compute the dissimilarity matrix required for MDS computation
        dissimilarity_metrics = ['euclidean', 'cityblock', 'mahalanobis', 'seuclidean', 'minkowski', 'chebyshev',
                                 'cosine', 'correlation', 'spearman', 'hamming', 'jaccard']
        dij_metric = self.dissimilarity_metric.lower()

        if dij_metric in dissimilarity_metrics:
            dij = pdist1(self.df.values, dij_metric)
            dij_matrix: None = distance.squareform(dij)

        else:
            print("Use a dissimilarity metric present in pdist1 from pydist2 package")

        for i in range(0, self.num_realizations):
            embedding_subset = MDS(dissimilarity='precomputed', n_components=2, n_init=20, max_iter=1000,
                                   random_state=random_seeds[i])
            mds_transformed_subset = embedding_subset.fit_transform(dij_matrix)

            if normalize_projections:
                scaler = StandardScaler()
                mds_transformed_subset = scaler.fit_transform(mds_transformed_subset)

            raw_stress = embedding_subset.stress_
            dissimilarity_matrix = embedding_subset.dissimilarity_matrix_
            stress_1 = np.sqrt(raw_stress / (0.5 * np.sum(dissimilarity_matrix ** 2)))
            norm_stress.append(stress_1)  # [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
            mds1.append(mds_transformed_subset[:, 0])
            mds2.append(mds_transformed_subset[:, 1])

            if self.dim_projection == '2D':  # i.e., if LDS is 2D
                real_i = np.column_stack((mds1[i], mds2[i]))  # stack projections for all realizations
            elif self.dim_projection == '3D':  # i.e., if LDS is 3D
                real_i = np.column_stack((mds1[i], mds2[i], [0] * len(mds1[i])))  # stack projections for all realizations
            else:
                raise TypeError("Use an LDS projection of '2D' or '3D' as dim_projection variable input in class.")

            all_real.append(real_i)

        # Make the LD space invariant to  translation, rotation, reflection/flipping, This applies the proposed
        # method to all realization and the base case individually to yield a unique solution.

        for i in range(1, len(all_real)):
            # Recover the rotation and translation matrices, R,T respectively for each realization

            if self.dim_projection == '2D':  # i.e., if LDS is 2D
                ret_R, ret_T = rigid_transform_2D(np.transpose(all_real[i]), np.transpose(all_real[0]))
                t.append(ret_T)
                r.append(ret_R)
                # Compare the recovered R and T with the base case by creating a new coordinate scheme via prior
                # solutions of r, and t
                new_coord = (ret_R @ np.transpose(all_real[i])) + np.expand_dims(ret_T, axis=1)
                calc_real.append(new_coord)

            elif self.dim_projection == '3D':  # i.e., if LDS is 3D
                ret_R, ret_T = rigid_transform_3D(np.transpose(all_real[i]), np.transpose(all_real[0]))
                t.append(ret_T)
                r.append(ret_R)
                # Compare the recovered R and T with the base case by creating a new coordinate scheme via prior
                # solutions of r, and t
                new_coord = (ret_R @ np.transpose(all_real[i])) + ret_T
                calc_real.append(new_coord)

            # Find the rmse as an error check between corrected realization and base case
            rmse_err = rmse(new_coord, all_real[0])
            all_rmse.append(rmse_err)

        # update
        self.random_seeds = random_seeds
        self.all_real = all_real
        self.calc_real = calc_real
        self.all_rmse = all_rmse
        self.norm_stress = norm_stress

        return random_seeds, all_real, calc_real, all_rmse, norm_stress

    def real_plotter(self, response, r_idx, Ax, Ay, title, x_off, y_off, cmap, array2=None,
                     annotate=True, save=True):
        """

        Parameters: TODO
        ----------
        idx
        response
        r_idx
        random_seeds
        Ax
        Ay
        title
        x_off
        y_off
        cmap
        array2
        save

        Returns
        -------

        """

        if self.all_real is None:
            raise TypeError("Run rung_rigid_MDS first.")
        # Basis for automated subplot
        num_cols = 2
        subplot_nos = len(r_idx)
        if subplot_nos % num_cols == 0:
            num_rows = subplot_nos // num_cols
        else:
            num_rows = (subplot_nos // num_cols) + 1

        # Make Plots
        plt.figure()

        if array2 is None:
            for i in range(0, len(r_idx)):
                ax = plt.subplot(num_rows, num_cols, i + 1)
                pairplot = sns.scatterplot(x=self.all_real[i][:, 0], y=self.all_real[i][:, 1],
                                           hue=self.df_idx[response], s=60, markers='o',
                                           palette=cmap, edgecolor="black", ax=ax)
                pairplot.set_xlabel(Ax)
                pairplot.set_ylabel(Ay)
                pairplot.set_title(title[i] + str(r_idx[i]) + " at seed " + str(self.random_seeds[i]))

                if annotate:
                    for j, txt in enumerate(self.df_idx[self.idx]):
                        pairplot.annotate(txt, (self.all_real[i][:, 0][j] + x_off, self.all_real[i][:, 1][j] + y_off),
                                          size=10, style='italic')

        else:
            for k in range(1, len(r_idx)):
                ax = plt.subplot(num_rows, num_cols, k)
                pairplot = sns.scatterplot(x=self.calc_real[k][0], y=self.calc_real[k][1], hue=self.df_idx[response],
                                           s=60, markers='o', palette=cmap, edgecolor="black", ax=ax)
                pairplot.set_xlabel(Ax)
                pairplot.set_ylabel(Ay)
                pairplot.set_title("Stabilized solution for " + title[k].lower() + str(r_idx[k]) + " at seed " +
                                   str(self.random_seeds[k - 1]))

                if annotate:
                    for index_, txt in enumerate(self.df_idx[self.idx]):
                        pairplot.annotate(txt, (self.calc_real[k][0][index_] + x_off, self.calc_real[k][1][index_] + y_off), size=10,
                                          style='italic')

            # Add base case to subplot for direct comparison of stabilized solution obtained
            ax = plt.subplot(num_rows, num_cols, k + 1)
            pairplot = sns.scatterplot(x=self.all_real[0][:, 0], y=self.all_real[0][:, 1], hue=self.df_idx[response],
                                       s=60, markers='o', palette=cmap, edgecolor="black", ax=ax)
            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title(title[0] + str(r_idx[0]) + " at seed " + str(self.random_seeds[0]))

            if annotate:
                for index_, txt in enumerate(self.df_idx[self.idx]):
                    pairplot.annotate(txt, (self.all_real[0][:, 0][index_] + x_off, self.all_real[0][:, 1][index_] + y_off),
                                      size=10, style='italic')

        # Figure info
        ax.set_aspect('auto')
        ax.legend(fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Aesthetics
        plt.subplots_adjust(left=0.0, bottom=0.0, right=2., top=2., wspace=0.3, hspace=0.3, )
        if save:
            plt.savefig('Variations with seeds 2x2 for data subset with tracking.tiff', dpi=300, bbox_inches='tight')
        plt.show()


    def bivariate_plotter(self, palette_, response, x_off, y_off, title, plot_type, Ax, Ay, annotate=True, save=True):
        """

        Parameters
        ----------
        x_off
        y_off
        annotate
        palette_
        response
        title
        plot_type
        Ax
        Ay
        save

        Returns
        -------

        """
        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")

        mds1_vec = None
        mds2_vec = None
        df = self.df.copy(deep=True)
        plot_type = plot_type.lower()

        # Palette assignment
        if palette_ == 1:
            palette_ = sns.color_palette("rocket_r", n_colors=len(np.unique(self.df_idx[response].values)) + 1)

        elif palette_ == 2:
            palette_ = sns.color_palette("bright", n_colors=len(np.unique(self.df_idx[response].values)) + 1)

        else:
            palette_ = None

        # Plot_type assignment
        if plot_type == 'variation':
            for i in range(0, len(self.all_real)):
                mds1_vec = self.all_real[i][:, 0]
                mds2_vec = self.all_real[i][:, 1]
                pairplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, hue=self.df_idx[response], s=60, markers='o',
                                           alpha=0.1,
                                           palette=palette_, edgecolor="black", legend=False)

                pairplot.set_xlabel(Ax)
                pairplot.set_ylabel(Ay)
                pairplot.set_title(title)

                if save:
                    plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

        elif plot_type == 'jitters':
            for i in range(0, len(self.calc_real)):
                mds1_vec = np.transpose(self.calc_real[i][0, :])
                mds2_vec = np.transpose(self.calc_real[i][1, :])

                pairplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, hue=self.df_idx[response], s=60, markers='o',
                                           alpha=0.1,
                                           palette=palette_, edgecolor="black", legend=False)
            if annotate:
                for index, label in enumerate(range(1, len(mds1_vec) + 1)):
                    plt.annotate(label, (mds1_vec[index] + x_off, mds2_vec[index] + y_off), size=8,
                                 style='italic')

            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title(title)

            if save:
                plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

        elif plot_type == 'uncertainty':
            # Plot realizations as a scatter plot
            for i in range(0, len(self.calc_real)):
                mds1_vec = np.transpose(self.calc_real[i][0, :])
                mds2_vec = np.transpose(self.calc_real[i][1, :])
                if i == 0:
                    pairplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, s=30, markers='o',
                                               alpha=0.3, edgecolor="black", linewidths=2,
                                               palette=palette_, hue=self.df_idx[response])
                else:
                    pairplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, s=30, markers='o', palette=palette_,
                                               alpha=0.1, legend=False, hue=self.df_idx[response])

            # Expectation of stabilized solution over all realizations for each sample
            array_exp = np.mean(self.calc_real, axis=0)

            # Plot the expectation of all realizations on the scatter plot
            sns.scatterplot(x=array_exp[0, :], y=array_exp[1, :], s=25, marker='x', linewidths=4,
                            alpha=1, color='k', edgecolor="black", label='expectation',
                            legend=True)

            if annotate:
                for index, label in enumerate(range(1, len(array_exp[0, :]) + 1)):
                    plt.annotate(label, (array_exp[0, :][index] + x_off, array_exp[1, :][index] + y_off), size=8,
                                 style='italic')

            # Plot lines between each data point in realizations and its corresponding value in the
            # expectation array
            # for j in range(0, len(self.calc_real)):
            #     for i in range(array_exp.shape[1]):
            #         if i == 0 and j == 0:
            #             plt.plot([np.transpose(self.calc_real[j][0, i]), array_exp[0, i]],
            #                      [np.transpose(self.calc_real[j][1, i]), array_exp[1, i]],
            #                      'r-', lw=2, alpha=0.1, label='sample perturbation')
            #         else:
            #             plt.plot([np.transpose(self.calc_real[j][0, i]), array_exp[0, i]],
            #                      [np.transpose(self.calc_real[j][1, i]), array_exp[1, i]],
            #                      'r-', lw=2, alpha=0.1)

            # Aesthetics
            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title(title)
            plt.legend()
            if save:
                plt.savefig('2D registration jitters uncertainty w.r.t expectation for all realizations.tiff',
                            dpi=300, bbox_inches='tight')
            plt.show()

        else:
            print("Use a plot_type of `variation`, `jitters`, or `uncertainty`")

    def expectation(self, r_idx, Ax, Ay, verbose=False):
        """
        expectation of all the calc_real
        Parameters: TODO
        ----------
        r_idx: changed to a 0
        base case realization index
        Ax:
        Ay:
        verbose:

        Returns
        -------

        """
        if self.all_real is None:
            raise TypeError("Run rung_rigid_MDS first.")

        # Base case
        sig_x = np.var(self.all_real[r_idx][:, 0])
        sig_y = np.var(self.all_real[r_idx][:, 1])
        sig_eff = sig_x + sig_y

        # Expectation of stabilized solution over all realizations for each sample
        E = np.mean(self.calc_real, axis=0)
        sigma_x = np.var(E[0, :])
        sigma_y = np.var(E[1, :])
        sigma_eff = sigma_x + sigma_y

        if verbose:
            print("The effective variance of the base case is", round(sig_eff, 4), "with a " + Ax + " variance of",
                  round(sig_x, 4), " and " + Ay + " variance of", round(sig_y, 4))
            print("\nThe effective variance of the expected stabilized solution is", round(sigma_eff, 4),
                  "with a " + Ax +
                  " variance of", round(sigma_x, 4), " and " + Ay + " variance of", round(sigma_y, 4))

        # Update
        self.array_exp = E
        return E

    def expect_plotter(self, r_idx, Lx, Ly, xmin, xmax, ymin, ymax, save=True):
        """

        Parameters:TODO
        ----------
        array_exp
        r_idx
        Lx
        Ly
        xmin
        xmax
        ymin
        ymax
        save

        Returns
        -------

        """
        if self.all_real is None:
            raise TypeError("Run rung_rigid_MDS first.")
        # For x-direction projection in LD space
        sns.kdeplot(self.all_real[r_idx][:, 0], label='Base case ' + Lx, color='blue')
        sns.kdeplot(self.array_exp[0, :], label=Lx + ' stabilized expectation', color='magenta', alpha=0.4)

        # For y-direction projection in LD space
        sns.kdeplot(self.all_real[r_idx][:, 1], label='Base case ' + Ly, color='green')
        sns.kdeplot(self.array_exp[1, :], label=Ly + ' stabilized expectation', color='orange', alpha=0.4)

        # Aesthetics
        plt.legend(loc="best", fontsize=14)
        plt.xlabel('Projections', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.subplots_adjust(left=0.0, bottom=0.5, right=1.2, top=2.0, wspace=0.25, hspace=0.3)
        if save:
            plt.savefig('Comparisons for projections between stabilized solutions and base case distributions.tiff',
                        dpi=300,
                        bbox_inches='tight')
        plt.show()
        return

    def compare_plot(self, response, r_idx, Ax, Ay, x_off, y_off, cmap, annotate=True, save=True):
        """

        Parameters:TODO
        ----------
        idx
        response
        r_idx
        Ax
        Ay
        x_off
        y_off
        cmap
        save

        Returns
        -------

        """
        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")
        if self.array_exp is None:
            raise TypeError("Run expectation first.")
        plt.subplot(121)
        pairplot = sns.scatterplot(x=self.all_real[r_idx][:, 0], y=self.all_real[r_idx][:, 1],
                                   hue=self.df_idx[response],
                                   s=60, markers='o', palette=cmap, edgecolor="black")

        if annotate:
            for i, txt in enumerate(self.df_idx[self.idx]):
                pairplot.annotate(txt, (self.all_real[r_idx][:, 0][i] + x_off, self.all_real[r_idx][:, 1][i] + y_off),
                                  size=10, style='italic')
        pairplot.set_xlabel(Ax)
        pairplot.set_ylabel(Ay)
        pairplot.set_title("Base case realization at seed " + str(self.random_seeds[r_idx]))

        plt.subplot(122)
        pairplot = sns.scatterplot(x=self.array_exp[0, :], y=self.array_exp[1, :], hue=self.df_idx[response], s=60,
                                   markers='o', palette=cmap, edgecolor="black")

        if annotate:
            for i, txt in enumerate(self.df_idx[self.idx]):
                pairplot.annotate(txt, (self.array_exp[0, :][i] + x_off, self.array_exp[1, :][i] + y_off), size=10,
                                  style='italic')
        pairplot.set_xlabel(Ax)
        pairplot.set_ylabel(Ay)
        pairplot.set_title(
            "Expectation of Stabilized Solutions over " + str(self.num_realizations) + " realizations")

        plt.subplots_adjust(left=0.0, bottom=0.0, right=2., top=1., wspace=0.3, hspace=0.3, )
        if save:
            plt.savefig('Stabilized independent result vs expectation of stabilized results.tiff', dpi=300,
                        bbox_inches='tight')
        plt.show()

    def visual_model_check(self, norm_type, fig_name, array, expectation_compute=True, save=True):
        """

        Parameters:TODO
        ----------
        norm_type
        fig_name
        array
        expectation_compute
        save

        Returns
        -------

        """
        dists, projected_dists = None, None
        # Obtain dataframe with the standardized predictor features
        if expectation_compute is True:
            # Grab the expectation of the stabilized solution
            stabilized_expected_proj = np.transpose(array[:2, :])
        else:
            stabilized_expected_proj = array.copy()

        # insert distortion visual
        norm_type = norm_type.upper()

        if norm_type == 'L2':
            dists = manhattan_distances(self.df).ravel()
            nonzero = dists != 0  # select only non-identical samples pairs
            dists = dists[nonzero]
            projected_dists = manhattan_distances(stabilized_expected_proj).ravel()[nonzero]

        elif norm_type == 'L1':
            dists = euclidean_distances(self.df, squared=False).ravel()
            nonzero = dists != 0  # select only non-identical samples pairs
            dists = dists[nonzero]
            projected_dists = euclidean_distances(stabilized_expected_proj, squared=False).ravel()[nonzero]

        plt.subplot(221)
        plt.scatter(dists, projected_dists, c='red', alpha=0.2, edgecolor='black')
        plt.arrow(0, 0, 200, 200, width=0.02, color='black', head_length=0.0, head_width=0.0)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.xlabel("Pairwise Distance: original space")
        plt.ylabel("Pairwise Distance: projected space")
        plt.title("Pairwise Distance: Projected to 2 components")

        rates = projected_dists / dists
        print("Distance Ratio, mean: %0.4f, standard deviation %0.4f." % (np.mean(rates), np.std(rates)))

        plt.subplot(222)
        plt.hist(rates, bins=50, range=(0.5, 1.5), color='red', alpha=0.2, edgecolor='k')
        plt.xlabel("Distance Ratio: projected / original")
        plt.ylabel("Frequency")
        plt.title("Pairwise Distance: Projected to 2 Components")

        plt.subplot(223)
        plt.hist(dists, bins=50, range=(0., 15.), color='red', alpha=0.2, edgecolor='k')
        plt.xlabel("Pairwise Distance")
        plt.ylabel("Frequency")
        plt.title("Pairwise Distance: Original Data")

        plt.subplot(224)
        plt.hist(projected_dists, bins=50, range=(0., 15.), color='red', alpha=0.2, edgecolor='k')
        plt.xlabel("Pairwise Distance")
        plt.ylabel("Frequency")
        plt.title("Pairwise Distance: Projected to 2 Components")

        # Aesthetics
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.7, top=2.3, wspace=0.2, hspace=0.3)
        if save:
            plt.savefig(fig_name + '.tiff', dpi=300, bbox_inches='tight')
        plt.show()

    # noinspection PyTypeChecker
    @staticmethod
    def convex_hull(array, title, x_off, y_off, Ax, Ay, expectation_compute=True, make_figure=True, n_case=True,
                    annotate=True, save=True):
        # Using samples from either n-case or n+1 case scenario to make a convex hull i.e., convex polygon

        if expectation_compute is True:
            my_points = np.transpose(array[:2, :])
        else:
            my_points = array[0][:, :2]  # all samples in projected space

        hull = ConvexHull(my_points)

        # Check for point in polygon
        vertices = my_points[hull.vertices]  # the anchors as an array
        polygon = Polygon(vertices)

        # check if data is a strict convex polygon
        binary_bool = is_convex_polygon(polygon)
        if binary_bool is False:
            return "Convex polygon assumption not met, do not use this workflow"

        else:

            if make_figure:
                if n_case is True:
                    # Make sample point visuals for N samples case
                    plt.scatter(my_points[:, 0], my_points[:, 1], marker='o', s=50, color='white', edgecolors="black")


                else:
                    # Make sample point visuals of added sample in N+1 samples case
                    plt.scatter(my_points[:-1, 0], my_points[:-1, 1], marker='o', s=50, color='white', edgecolors="black")
                    plt.scatter(my_points[-1, 0], my_points[-1, 1], marker='*', s=90, color='black', edgecolors="black")

                if annotate:
                    # Annotate sample index
                    for index, label in enumerate(range(1, len(my_points[:, 0]) + 1)):
                        plt.annotate(label, (my_points[:, 0][index] + x_off, my_points[:, 1][index] + y_off), size=8,
                                     style='italic')

                # Make figure to visualize convex hull polygon and highlight polygon formed
                for simplex in hull.simplices:
                    plt.plot(my_points[simplex, 0], my_points[simplex, 1], 'r--')  # k-
                    plt.fill(my_points[hull.vertices, 0], my_points[hull.vertices, 1], c='yellow', alpha=0.01)

                # Aesthetics
                plt.title(title)
                plt.xlabel(Ax)
                plt.ylabel(Ay)
                if save:
                    plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

                plt.show()
            return my_points, hull, vertices

    def marginal_dbn(self, save=True):
        """

        Returns:TODO
        -------

        """
        ns_features = self.df.columns.tolist()

        N = 10
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        for feat in range(len(ns_features)):
            x = self.df[ns_features[feat]]
            stdev = np.std(x)
            mean = np.mean(x)
            ax = sns.kdeplot(x, fill=True, ax=axs[feat])

            for i in range(1, 4):  # st. dev away from mean needed for visuals.
                x1 = np.linspace(mean - i * stdev, mean - (i - 1) * stdev, N)
                x2 = np.linspace(mean - (i - 1) * stdev, mean + (i - 1) * stdev, N)
                x3 = np.linspace(mean + (i - 1) * stdev, mean + i * stdev, N)
                x = np.concatenate((x1, x2, x3))
                x = np.where((mean - (i - 1) * stdev < x) & (x < mean + (i - 1) * stdev), np.nan, x)
                y = norm.pdf(x, mean, stdev)
                ax.fill_between(x, y, alpha=0.5)

            # Aesthetics
            axs[feat].set_xlabel('Normal scores for ' + ns_features[feat][3:])
            axs[feat].set_xticks(ticks=np.arange(-5, 5))
            axs[feat].set_ylabel("Marginal probability density")

        # Create legend with patch, color match using CSS colors in matplotlib
        std_1 = mpatches.Patch(color='sandybrown', label='+/- 1'r'$\sigma$')
        std_2 = mpatches.Patch(color='darkseagreen', label='+/- 2'r'$\sigma$')
        std_3 = mpatches.Patch(color='indianred', label='+/- 3'r'$\sigma$')
        plt.legend(handles=[std_1, std_2, std_3])
        plt.subplots_adjust(wspace=0.3)
        if save:
            plt.savefig('Marginal distributions and st. deviation thresholds for all predictors.tiff', dpi=300,
                        bbox_inches='tight')
        plt.show()


# noinspection PyUnboundLocalVariable
class RigidTransf_NPlus(RigidTransformation):
    def __init__(self, df, features, idx, num_realizations, base_seed, start_seed, stop_seed, dissimilarity_metric,
                 dim_projection):
        super().__init__(df, features, idx, num_realizations, base_seed, start_seed, stop_seed, dissimilarity_metric,
                         dim_projection)
        self.anchors1 = None
        self.anchors1 = None
        self.anchors2 = None
        self.R_anchors = None
        self.t_anchors = None
        self.rmse_err_anchors = None
        self.stable_coords_anchors = None
        self.stable_coords_alldata = None
        self.common_vertices_index = None
        self.common_vertices2_index = None


    def stabilize_anchors(self, array1, array2, hull_1, hull_2, normalize_projections=True):
        # Obtain the anchor points for n and n+1 scenarios
        vertices_index = hull_1.vertices
        vertices2_index = hull_2.vertices

        # Find the common anchor points/vertices between anchors in N and N+1 sample case
        common_indexes = np.isin(vertices_index, vertices2_index)

        # Extract the common indexes from vertices_index and vertices2_index
        common_vertices_index = np.intersect1d(vertices_index, vertices2_index)
        common_vertices2_index = np.intersect1d(vertices2_index, vertices_index)

        # Access the corresponding anchor points using the common indexes
        case1_anchors = array1[common_vertices_index]
        case2_anchors = array2[common_vertices2_index]

        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            anchors1 = np.column_stack((case1_anchors[:, 0], case1_anchors[:, 1]))
            anchors2 = np.column_stack((case2_anchors[:, 0], case2_anchors[:, 1]))
        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            anchors1 = np.column_stack((case1_anchors[:, 0], case1_anchors[:, 1], [0] * len(case1_anchors)))
            anchors2 = np.column_stack((case2_anchors[:, 0], case2_anchors[:, 1], [0] * len(case2_anchors)))
        else:
            raise TypeError("Use an LDS projection of '2D' or '3D' as dim_projection variable input in class.")

        # Recover the rotation and translation matrices R,t, respectively for the stable anchor points in n+1 to
        # match anchors in the n-case scenario
        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            R_anchors, t_anchors = rigid_transform_2D(np.transpose(anchors2), np.transpose(anchors1))
            # Compare the recovered R and t with the original by creating a new coordinate scheme via prior solutions
            # of R, t
            new_coord_anchors = (R_anchors @ np.transpose(anchors2)) + np.expand_dims(t_anchors, axis=1)
            # R_anchors_, t_anchors_ = rigid_transform_2D(new_coord_anchors, np.transpose(anchors1))
            # new_coord_anchors_ = (R_anchors_ @ new_coord_anchors) + np.expand_dims(t_anchors_, axis=1)

        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            R_anchors, t_anchors = rigid_transform_3D(np.transpose(anchors2), np.transpose(anchors1))

            # Compare the recovered R and t with the original by creating a new coordinate scheme via prior solutions
            # of R, t
            new_coord_anchors = (R_anchors @ np.transpose(anchors2)) + t_anchors
            # R_anchors_, t_anchors_ = rigid_transform_3D(new_coord_anchors, np.transpose(anchors1))
            # new_coord_anchors_ = (R_anchors_ @ new_coord_anchors) + t_anchors_

        # Find the rmse as an error check between estimated anchor points in n+1 scenario and anchor points in
        # n-scenario
        rmse_err_anchors = rmse(new_coord_anchors, anchors1)

        # Create a convex hull polygon of the normalized stabilized anchor points. Set this as an assertion!
        stable_coords_anchors = np.transpose(new_coord_anchors[:2, :])

        if normalize_projections:
            scaler = StandardScaler()
            stable_coords_anchors = scaler.fit_transform(stable_coords_anchors)

        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            anchors1 = np.column_stack((case1_anchors[:, 0], case1_anchors[:, 1]))
            anchors2 = np.column_stack((case2_anchors[:, 0], case2_anchors[:, 1]))
            stable_anchors_array = np.column_stack((array2[:, 0], array2[:, 1]))

            # Use the R and t matrix from the stabilized anchor solution and apply it to all samples in the n+1 scenario
            # to obtain the now stabilized solution for every sample point.
            new_coords_alldata = R_anchors@np.transpose(stable_anchors_array) + np.expand_dims(t_anchors, axis=1)

            # # Computationally heavier method, better to use above anchor registration method as proposed. To be used
            # when there is no SVD rigid transformation possible due to deformation of points and OOSP from the tails.
            # stable_anchors_array = np.column_stack((array2[:len(array2) - 1, 0], array2[:len(array2) - 1, 1]))
            # R_all, t_all = rigid_transform_2D(np.transpose(stable_anchors_array), np.transpose(array1))
            # new_coords_alldata = (R_all @ np.transpose(array2)) + np.expand_dims(t_all, axis=1)

        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            anchors1 = np.column_stack((case1_anchors[:, 0], case1_anchors[:, 1], [0] * len(case1_anchors)))
            anchors2 = np.column_stack((case2_anchors[:, 0], case2_anchors[:, 1], [0] * len(case2_anchors)))
            stable_anchors_array = np.column_stack((array2[:len(array2), 0], array2[:len(array2), 1], [0] * (len(array2))))

            # Use the R and t matrix from the stabilized anchor solution and apply it to all samples in the n+1 scenario
            # to obtain the now stabilized solution for every sample point.
            new_coords_alldata = (R_anchors @ np.transpose(stable_anchors_array)) + t_anchors

            # # Computationally heavier method, better to use above anchor registration method as proposed. To be used
            # when there is no SVD rigid transformation possible due to deformation of points and OOSP from the tails
            # stable_anchors_array = np.column_stack(
            #     (array2[:len(array2) - 1, 0], array2[:len(array2) - 1, 1], [0] * (len(array2) - 1)))
            # R_all, t_all = rigid_transform_3D(np.transpose(stable_anchors_array), np.transpose(array1))
            # new_coords_alldata = (R_all @ np.transpose(array2)) + t_all

        stable_coords_alldata = np.transpose(new_coords_alldata[:2, :])

        # Find the rmse as an error check between estimated stabilized points for all data in N+1 scenario and base case
        # in N-sample scenario
        rmse_err_alldata = rmse(new_coords_alldata, array2)

        # Update
        self.anchors1 = anchors1
        self.anchors2 = anchors2
        self.R_anchors = R_anchors
        self.t_anchors = t_anchors
        self.rmse_err_anchors = rmse_err_anchors
        self.stable_coords_anchors = stable_coords_anchors
        self.stable_coords_alldata = stable_coords_alldata
        self.common_vertices_index = common_vertices_index + 1  # +1 accounts for Python's indexing starting from 0
        self.common_vertices2_index = common_vertices2_index + 1  # +1 accounts for Python's indexing starting from 0
        return anchors1, anchors2, R_anchors, t_anchors, rmse_err_anchors, stable_coords_anchors, stable_coords_alldata,\
               rmse_err_alldata

    def stable_anchor_visuals(self, Ax, Ay, x_off, y_off, annotate=True, save=True):
        # Visualization of base case and stabilized solution
        fig, [ax0, ax1, ax2] = plt.subplots(1, 3)

        # For base case anchors i.e. in N case
        ax0.scatter(self.anchors1[:, 0], self.anchors1[:, 1], marker='o', s=50, color='blue', edgecolors="black")

        if annotate:
            for index, label in enumerate(self.common_vertices_index):
                ax0.annotate(label, (self.anchors1[:, 0][index] + x_off, self.anchors1[:, 1][index] + y_off),
                             size=10, style='italic')
        ax0.set_aspect('auto')
        ax0.set_title('Anchors from N sample case', size=14)
        ax0.set_xlabel(Ax, size=14)
        ax0.set_ylabel(Ay, size=14)
        ax0.tick_params(axis='both', which='major', labelsize=12)

        # For the realization anchors at N+1 case
        ax1.scatter(self.anchors2[:, 0], self.anchors2[:, 1], marker='o', s=50, color='blue', edgecolors="black")

        if annotate:
            for index, label in enumerate(self.common_vertices2_index):
                ax1.annotate(label, (self.anchors2[:, 0][index] + x_off, self.anchors2[:, 1][index] + y_off),
                             size=10, style='italic')
        ax1.set_aspect('auto')
        ax1.set_title('Anchors from N+1 sample case', size=14)
        ax1.set_xlabel(Ax, size=14)
        ax1.set_ylabel(Ay, size=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Visualize the normalized stabilized anchor points
        ax2.scatter(self.stable_coords_anchors[:, 0], self.stable_coords_anchors[:, 1], marker='o', s=50, color='blue',
                    edgecolors="black")

        if annotate:
            for index, label in enumerate(self.common_vertices_index):
                ax2.annotate(label, (self.stable_coords_anchors[:, 0][index] + x_off,
                                     self.stable_coords_anchors[:, 1][index] + y_off), size=10, style='italic')
        ax2.set_aspect('auto')
        ax2.set_title('Stabilized anchor solution', size=14)
        ax2.set_xlabel(Ax, size=14)
        ax2.set_ylabel(Ay, size=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)

        plt.subplots_adjust(left=0.0, bottom=0.0, right=3., top=1.3, wspace=0.25, hspace=0.3)
        if save:
            plt.savefig('Anchor sets & Stabilized Anchor set Solution.tiff', dpi=300, bbox_inches='tight')
        plt.show()
        return

    def stable_representation(self, title, Ax, Ay, x_off, y_off, sample_added, annotate=True, save=True):
        # Visualize the n+1 case for all samples with stabilized representation as obtained in the n-case implying
        # rotation, translation, and reflection invariance.
        plt.scatter(self.stable_coords_alldata[:sample_added - 1, 0], self.stable_coords_alldata[:sample_added - 1, 1],
                    marker='o', s=50, color='white', edgecolors="black")
        plt.scatter(self.stable_coords_alldata[sample_added - 1, 0], self.stable_coords_alldata[sample_added - 1, 1],
                    marker='*', color='k', s=90)

        if annotate:
            for index, label in enumerate(range(1, len(self.stable_coords_alldata[:, 0]) + 1)):
                plt.annotate(label, (self.stable_coords_alldata[:, 0][index] + x_off,
                                     self.stable_coords_alldata[:, 1][index] + y_off), size=8, style='italic')

        plt.title(title)
        plt.xlabel(Ax)
        plt.ylabel(Ay)
        if save:
            plt.savefig('Stabilized N+1 case with same representation as N case.tiff', dpi=300, bbox_inches='tight')
        plt.show()
        return

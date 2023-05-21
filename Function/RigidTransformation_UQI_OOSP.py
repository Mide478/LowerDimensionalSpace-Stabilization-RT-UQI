import math
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from pydist2.distance import pdist1
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.stats import norm
from shapely.geometry import Polygon
from sklearn.manifold import MDS  # multidimensional scaling
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, pairwise_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# TURN OFF ALL GRIDS either via sns or plt.
# noinspection PyTypeChecker
sns.set_style("whitegrid", {'axes.grid': False})


def matrix_scatter(dataframe, feat_title, left_adj, bottom_adj, right_adj, top_adj, wspace, hspace, title, palette_,
                   hue_=None, num_OOSP=None, n_case=True, save=True):
    """
    This function plots the matrix scatter plot for the given data.

    Arguments
    ---------
    dataframe : pandas DataFrame
        The input data.
    feat_title : list of str
        Column names of the predictor features.
    left_adj : float
        Left placement adjustment of the scatter plot.
    bottom_adj : float
        Bottom placement adjustment of the scatter plot.
    right_adj : float
        Right placement adjustment of the scatter plot.
    top_adj : float
        Top placement adjustment of the scatter plot.
    wspace : float
        Width placement adjustment of the scatter plot.
    hspace : float
        Height placement adjustment of the scatter plot.
    title : str
        Name of the figure.
    palette_ : int
        Integer representing the color palette to use.
    hue_ : str, optional
        Variable used to color the matrix scatter plot.
    num_OOSP : int, optional
        Number of OOSP samples added within the 95% confidence interval
    n_case : bool, optional
        Determines if N_case visuals are used.
    save : bool, optional
        Determines if the plot is saved.

    Returns
    -------
    None
    """

    # Hue assignment
    if hue_ is not None:
        palette_ = sns.color_palette("rocket_r", n_colors=len(dataframe[hue_].unique()) + 1) if palette_ == 1 \
            else sns.color_palette("bright", n_colors=len(dataframe[hue_].unique()) + 1)

    else:
        palette_ = sns.color_palette("rocket_r") if palette_ == 1 else sns.color_palette("bright")

    # For N_case visuals
    if n_case:
        sns.pairplot(dataframe, vars=feat_title, markers='o', diag_kws={'edgecolor': 'black'},
                     plot_kws=dict(s=50, edgecolor="black", linewidth=0.5), hue=hue_, corner=True,
                     palette=palette_)

    else:
        # Define marker type for last datapoint i.e., the additional sample in N+1 case
        last_marker = '*'

        # Create pairplot
        fig = sns.pairplot(data=dataframe[:-num_OOSP], vars=feat_title, diag_kws={'edgecolor': 'black'},
                           plot_kws=dict(s=50, edgecolor="black", linewidth=0.5), hue=hue_, corner=True,
                           markers='o', palette=palette_)

        # Plot the OOSP's with a stars marker type in all subplots
        if num_OOSP is not None:
            last_data = dataframe.iloc[-num_OOSP:]
            for i, feat1 in enumerate(feat_title):
                for j, feat2 in enumerate(feat_title):
                    if i == j:
                        continue
                    ax = fig.axes[i, j]
                    if ax is not None:
                        for _, row in last_data.iterrows():
                            last_datapoint = row[[feat2, feat1]].values
                            hue_value = row[hue_]

                            if hue_value in ('low', 'med', 'high', 'vhigh'):
                                color_index = ['low', 'med', 'high', 'vhigh'].index(hue_value)
                                color = palette_[color_index]
                                ax.scatter(last_datapoint[0], last_datapoint[1], marker=last_marker, s=200, color=color,
                                           edgecolors="black", linewidth=0.5)

    plt.subplots_adjust(left=left_adj, bottom=bottom_adj, right=right_adj, top=top_adj, wspace=wspace, hspace=hspace)

    if save:
        plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


def make_levels(data, cat_response, num_response):
    """
    This function assigns ordinal levels to a numerical response variable based on predefined bins.

    Arguments
    ---------

    data : pandas DataFrame
        The input data.
    cat_response : str
        The column name for the new categorical response variable.
    num_response : str
        The column name for the numerical response variable.

    Returns
    -------
    DataFrame : The input DataFrame with the new categorical response variable added.

    """

    bins = [0, 2500, 5000, 7500, 10000]  # assign the production bins (these are the fence posts)
    labels = ['low', 'med', 'high', 'vhigh']  # assign the labels

    # Use pd.cut() to create the categorical response variable based on the numerical response and predefined bins
    data[cat_response] = pd.cut(data[num_response], bins, labels=labels)
    return data


def standardizer(dataset, features, keep_only_std_features=False):
    """
    Standardizes the selected features of a dataframe to have a mean of 0 and variance of 1.

    Arguments
    ---------
    dataset : pandas.DataFrame
        The input dataframe containing the features to be standardized.
    features : list of str
        The column names of the features to be standardized.
    keep_only_std_features : bool, optional
        Specifies whether to keep only the standardized features in the resulting dataframe.
        If False, the original dataset is returned with additional columns for the standardized features.
        If True, only the standardized features are returned to the resulting dataframe.
        Default is False.

    Returns
    -------
    df : pandas.DataFrame
        The standardized dataframe.

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


def normalizer(array):
    """
    Normalizes the values of an array to a specified range.

    Arguments
    ---------
    array : numpy.ndarray
        The input array to be normalized.

    Returns
    -------
    norrmalized_array : numpy.ndarray
        The normalized array.

    """

    scaler = MinMaxScaler(feature_range=(-4, 4))
    normalized_array = scaler.fit_transform(array)
    return normalized_array


def generate_random_seeds(seed, num_realizations, lower_bound, upper_bound):
    """
    Generates a list of random seeds.

    Arguments
    ---------
    seed : int
        The seed value for the random number generator.
    num_realizations : int
        The number of random seeds to generate.
    lower_bound : int
        The lower bound of the random seed range (inclusive).
    upper_bound : int
        The upper bound of the random seed range (inclusive).

    Returns
    -------
    random_seeds : list
        A list of randomly generated seeds.

    """

    random.seed(seed)
    random_seeds = [random.randint(lower_bound, upper_bound) for _ in range(num_realizations)]
    return random_seeds


def rigid_transform_2D(A, B, verbose=False):
    """
    Performs a rigid transformation (rotation and translation) on 2D point sets A and B.

    Arguments
    ---------
    A : numpy.ndarray
        2xN matrix of points.
    B : numpy.ndarray
        2xN matrix of points.
    verbose : bool, optional
        If True, prints additional information.

    Returns
    -------
    R : numpy.ndarray
        2x2 rotation matrix.
    t : numpy.ndarray
        2x1 translation vector.

    Raises
    ------
    ValueError
        If the input matrices A and B are not of shape (2, N).
    """

    if A.shape != B.shape or A.shape[0] != 2:
        raise ValueError("Matrices A and B must be of shape (2, N).")

    # Center the point sets
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    centered_A = A - np.expand_dims(centroid_A, axis=1)
    centered_B = B - np.expand_dims(centroid_B, axis=1)

    # Perform SVD on the centered point sets
    H = centered_A @ centered_B.T
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix R
    R = Vt.T @ U.T

    # Correct for special reflection case
    if np.linalg.det(R) < 0 and verbose:
        print("det(R) < 0, reflection detected! Solution corrected for it")
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # Calculate the translation vector t
    t = centroid_B - R @ centroid_A
    return R, t


def rigid_transform_3D(A, B, verbose=False):
    """
    Fits a rigid transform to a set of 3D points.

    Arguments
    ---------
    A : numpy.ndarray
        3xN matrix of points.
    B : numpy.ndarray
        3xN matrix of points.
    verbose : bool, optional
        If True, prints additional information.

    Returns
    -------
    R : numpy.ndarray
        3x3 rotation matrix.
    t : numpy.ndarray
        3x1 translation column vector.

    Raises
    ------
    ValueError
        If the input matrices A and B are not of shape (3, N).
    """

    if A.shape != B.shape or A.shape[0] != 3:
        raise ValueError("Matrices A and B must be of shape (3, N).")

    # Find the centroids (mean) of each point set
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # Center the points by subtracting the centroids
    Am = A - centroid_A
    Bm = B - centroid_B

    # Compute the covariance matrix
    H = Am @ np.transpose(Bm)

    # Perform SVD on the covariance matrix
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix R
    R = Vt.T @ U.T

    # Correct for special reflection case
    if np.linalg.det(R) < 0 and verbose:
        print("det(R) < 0, reflection detected! Solution corrected for it")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Calculate the translation vector t
    t = -R @ (centroid_A + centroid_B)
    return R, t


def is_convex_polygon(polygon):
    """
    Checks if the polygon defined by the sequence of 2D points is a strictly convex polygon.

    Arguments
    ---------
    polygon : list
        Sequence of 2D points representing the polygon.

    Returns
    -------
    bool
        True if the polygon is a strictly convex polygon, False otherwise.

    Notes
    -----
    - The algorithm checks if the points are valid, the side lengths are non-zero,
      the interior angles are strictly between zero and a straight angle, and
      the polygon does not intersect itself.
    - No explicit check is done for zero internal angles (180-degree direction-change angle)
      as this is covered in other ways, including the `n < 3` check.
    """

    TWO_PI = 2 * math.pi

    try:
        # Check for too few points
        if len(polygon) < 3:
            return False

        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0

        # Check each point (the side ending there, its angle) and accumulate angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)

            if old_x == new_x and old_y == new_y:
                return False  # Repeated consecutive points

            # Calculate and check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -math.pi:
                angle += TWO_PI  # Make it in the half-open interval (-Pi, Pi]
            elif angle > math.pi:
                angle -= TWO_PI

            if ndx == 0:  # If first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # If other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # Not both positive or both negative
                    return False

            # Accumulate the direction-change angle
            angle_sum += angle

        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1

    except (ArithmeticError, TypeError, ValueError):
        return True


def rmse(array1, array2):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays.

    Arguments
    ---------
    array1 : numpy.ndarray
        Array representing the recovered realization "i" array from R, T calculation.
    array2 : numpy.ndarray
        Array representing the base case.

    Returns
    -------
    rmse_error : float
        RMSE value
    """

    var1 = np.transpose(array1) - array2
    var1 = var1 * var1
    var1 = np.sum(var1)
    rmse_error = np.sqrt(var1 / len(array1[0, :]))
    return rmse_error


def make_sample_within_ci(dataframe, num_OOSP):
    """
    Sample a single row from a dataframe of multiple columns such that it is within a 95% confidence interval (CI).

    Arguments
    ---------
    dataframe : pandas DataFrame
        Original dataset with predictors to make OOSP
    num_OOSP : int
        Number of OOSP samples to add within the 95% confidence interval to the dataframe

    Returns
    -------
    data : pandas DataFrame
        DataFrame with a single row of sampled values from each column, such that each value falls within
        a 95% confidence interval of the original dataset.
    random_seed : list
        List of random states used to generate OOSP's
    """

    # Initialize list to store random seeds
    random_seeds = set()

    # Calculate mean, standard deviation, and bounds for each column
    n = len(dataframe)
    means = dataframe.mean()
    stds = dataframe.std()
    t = 1.96  # 95% confidence interval for a normal distribution
    lower_bounds, upper_bounds = means - t * (stds / np.sqrt(n)), means + t * (stds / np.sqrt(n))

    # Generate random values within 95% CI for each column
    samples = []
    while len(samples) < num_OOSP:
        random_seed = np.random.randint(0, 100000)
        if random_seed not in random_seeds:
            np.random.seed(random_seed)
            random_seeds.add(random_seed)

        # Generate random values within 95% CI for each column
        sample = np.random.uniform(lower_bounds, upper_bounds)
        samples.append(sample)

    # Combine sampled values into single row
    sampled_data = pd.DataFrame(samples, columns=dataframe.columns)

    # Add to dataframe
    data = dataframe.copy().append(sampled_data, ignore_index=True)
    return data, random_seeds


class RigidTransformation:
    def __init__(self, df, features, idx, num_OOSP, num_realizations, base_seed, start_seed, stop_seed, dissimilarity_metric,
                 dim_projection):
        """
        Initializes the RigidTransformation class.

        Arguments
        ---------
        df : pandas DataFrame
            The input DataFrame containing the data for rigid transformation.
        features : list
            A list of features column names to be used for transformation.
        idx : str
            The name of the index or UWI column in the DataFrame.
        num_OOSP : int, optional
            Number of OOSP samples added within the 95% confidence interval
        num_realizations : int
            The number of realizations to generate.
        base_seed : int
            The base seed value for random number generation.
        start_seed : int
            The starting seed value for realizations.
        stop_seed : int
            The stopping seed value for realizations.
        dissimilarity_metric : str
            User-specified dissimilarity metric to be used for the NDR i.e., MDS computation.
        dim_projection : str
            The dimension of the LDS projection (2D or 3D).
        """

        self.df_idx = df.copy()
        self.df = standardizer(df, features, keep_only_std_features=True)
        self.df_idx[idx] = np.arange(1, len(self.df) + 1).astype(int)
        self.num_OOSP = num_OOSP
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
        Runs the Rigid MDS algorithm for generating realizations.

        Arguments
        ---------
        normalize_projections : bool, optional
            Flag indicating whether to normalize the MDS projections. Default is True.

        Returns
        -------
        random_seeds : list
            List of random seeds used for each realization.
        all_real : list
            List of all realizations prepared for rigid transformation.
        calc_real : list
            List of analytical estimations of each realization, now stabilized from the R and T recovered via rigid
            transformation.
        all_rmse : list
            List of RMSE (Root Mean Square Error) between the corrected realization and the base case.
        norm_stress : list
            List of normalized stress values for each realization.
        """

        # Arrays below store random values for each realization
        random_seeds = generate_random_seeds(self.base_seed, self.num_realizations, self.start_seed, self.stop_seed)

        mds1 = []  # MDS projection 1
        mds2 = []  # MDS projection 2
        norm_stress = []
        all_real = []  # All realizations prepared for rigid transform
        t = []
        r = []
        all_rmse = []
        calc_real = []  # analytical estimation of each realization from R,T recovered

        dissimilarity_metrics = ["euclidean", "cityblock", "mahalanobis", "seuclidean", "minkowski", "chebyshev",
                                 "cosine", "correlation", "spearman", "hamming", "jaccard"]
        dij_metric = self.dissimilarity_metric.lower()

        if dij_metric in dissimilarity_metrics:
            dij = pdist1(self.df.values, dij_metric)
            dij_matrix: None = distance.squareform(dij)
        else:
            raise ValueError("Use a dissimilarity metric present in pdist1 from pydist2 package")

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
            norm_stress.append(stress_1)
            mds1.append(mds_transformed_subset[:, 0])
            mds2.append(mds_transformed_subset[:, 1])

            if self.dim_projection == '2D':
                real_i = np.column_stack((mds1[i], mds2[i]))
            elif self.dim_projection == '3D':
                real_i = np.column_stack((mds1[i], mds2[i], np.zeros(len(mds1[i]))))
            else:
                raise ValueError("Use an LDS projection of '2D' or '3D' as dim_projection variable input in class.")

            all_real.append(real_i)

        # Make LDS invariant to Euclidean transformations as proposed.
        for i in range(1, len(all_real)):
            # Recover the rotation and translation matrices, R,T respectively for each realization

            if self.dim_projection == '2D':  # i.e., if LDS is 2D
                ret_R, ret_T = rigid_transform_2D(np.transpose(all_real[i]), np.transpose(all_real[0]))
                t.append(ret_T)
                r.append(ret_R)
                new_coord = (ret_R @ np.transpose(all_real[i])) + np.expand_dims(ret_T, axis=1)
                calc_real.append(new_coord)

            elif self.dim_projection == '3D':  # i.e., if LDS is 3D
                ret_R, ret_T = rigid_transform_3D(np.transpose(all_real[i]), np.transpose(all_real[0]))
                t.append(ret_T)
                r.append(ret_R)
                new_coord = (ret_R @ np.transpose(all_real[i])) + ret_T
                calc_real.append(new_coord)

            rmse_err = rmse(new_coord, all_real[0])  # estimated RT realization vs base case
            all_rmse.append(rmse_err)

        #  Update
        self.random_seeds = random_seeds
        self.all_real = all_real
        self.calc_real = calc_real
        self.all_rmse = all_rmse
        self.norm_stress = norm_stress
        return random_seeds, all_real, calc_real, all_rmse, norm_stress


    def real_plotter(self, response, r_idx, Ax, Ay, title, x_off, y_off, cmap, array2=None, annotate=True, save=True):
        """
        Plots the realizations.

        Arguments
        ---------
        response: str
            The response feature to use for coloring the scatter plot.
        r_idx : list
            List of indices of realizations to plot.
        Ax : str
            Label for the x-axis.
        Ay: str
            Label for the y-axis.
        title : list
            List of titles for each subplot.
        x_off : float
            Offset value for x-axis annotations.
        y_off : float
            Offset value for y-axis annotations.
        cmap : str or colormap object
            Color map to use for coloring the scatter plot.
        array2 : None or ndarray, optional
            Array containing stabilized points per realization for comparison. Default is None.
        annotate : bool, optional
            Flag indicating whether to annotate the data points. Default is True.
        save : bool, optional
            Flag indicating whether to save the plot as an image. Default is True.

        Returns
        -------
        None
        """

        if self.all_real is None:
            raise TypeError("Run rung_rigid_MDS first.")

        subplot_nos = len(r_idx)
        num_cols = 2
        num_rows = (subplot_nos + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        axs = axs.flatten()

        if array2 is None:
            for i, ax in enumerate(axs):
                if i < subplot_nos:
                    realization_idx = r_idx[i]
                    pairplot = sns.scatterplot(x=self.all_real[realization_idx][:, 0],
                                               y=self.all_real[realization_idx][:, 1],
                                               hue=self.df_idx[response], s=60, markers='o', palette=cmap,
                                               edgecolor="black", ax=ax)
                    pairplot.set_xlabel(Ax)
                    pairplot.set_ylabel(Ay)
                    pairplot.set_title(title[i] + str(realization_idx) + " at seed " + str(self.random_seeds[i]))

                    if annotate:
                        for j, txt in enumerate(self.df_idx[self.idx]):
                            pairplot.annotate(txt,
                                              (self.all_real[realization_idx][:, 0][j] + x_off,
                                               self.all_real[realization_idx][:, 1][j] + y_off),
                                              size=10, style='italic')

        else:
            for i, ax in enumerate(axs[:-1]):
                if i < subplot_nos - 1:
                    realization_idx = r_idx[i]
                    pairplot = sns.scatterplot(x=self.calc_real[realization_idx][0],
                                               y=self.calc_real[realization_idx][1],
                                               hue=self.df_idx[response], s=60, markers='o', palette=cmap,
                                               edgecolor="black", ax=ax)
                    pairplot.set_xlabel(Ax)
                    pairplot.set_ylabel(Ay)
                    pairplot.set_title(
                        "Stabilized solution for " + title[i].lower() + str(realization_idx) + " at seed " +
                        str(self.random_seeds[i]))

                    if annotate:
                        for index_, txt in enumerate(self.df_idx[self.idx]):
                            pairplot.annotate(txt,
                                              (self.calc_real[realization_idx][0][index_] + x_off,
                                               self.calc_real[realization_idx][1][index_] + y_off),
                                              size=10, style='italic')

            # Add base case subplot for direct comparison of the stabilized solution obtained
            realization_idx = r_idx[subplot_nos - 1]
            ax = axs[subplot_nos - 1]
            pairplot = sns.scatterplot(x=self.all_real[0][:, 0], y=self.all_real[0][:, 1],
                                       hue=self.df_idx[response], s=60, markers='o', palette=cmap,
                                       edgecolor="black", ax=ax)
            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title(title[0] + str(realization_idx) + " at seed " + str(self.random_seeds[0]))

            if annotate:
                for index_, txt in enumerate(self.df_idx[self.idx]):
                    pairplot.annotate(txt,
                                      (self.all_real[0][:, 0][index_] + x_off, self.all_real[0][:, 1][index_] + y_off),
                                      size=10, style='italic')

        for ax in axs:
            ax.set_aspect('auto')
            ax.legend(fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=12)

        plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.5, wspace=0.2, hspace=0.2)

        if save:
            plt.savefig('Variations with seeds 2x2 for data subset with tracking.tiff', dpi=300, bbox_inches='tight')

        plt.show()


    def bivariate_plotter(self, palette_, response, x_off, y_off, title, plot_type, Ax, Ay, annotate=True, save=True):
        """
        Plots bivariate scatter plots based on the specified plot type.

        Arguments
        ---------
        palette_ : int
            Integer value indicating the color palette to use.
        response : str
            The response variable to use for coloring the scatter plot.
        x_off : float
            Offset value for x-axis annotations.
        y_off : float
            Offset value for y-axis annotations.
        title : str
            Title for the plot.
        plot_type : str
            Type of plot to create ('variation', 'jitters', or 'uncertainty').
        Ax : str
            Label for the x-axis.
        Ay : str
            Label for the y-axis.
        annotate : bool, optional
            Flag indicating whether to annotate the data points. Default is True.
        save : bool, optional
            Flag indicating whether to save the plot as an image. Default is True.

        Returns
        -------
        None
        """

        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")

        mds1_vec = None
        mds2_vec = None
        df = self.df.copy(deep=True)

        #  Palette assignment
        if palette_ == 1:
            palette_ = sns.color_palette("rocket_r", n_colors=len(np.unique(self.df_idx[response].values)))
        elif palette_ == 2:
            palette_ = sns.color_palette("bright", n_colors=len(np.unique(self.df_idx[response].values)))
        else:
            palette_ = None

        #  Plot_type assignment
        if plot_type.lower() == 'variation':
            for i, real in enumerate(self.all_real):
                mds1_vec = real[:, 0]
                mds2_vec = real[:, 1]
                scatterplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, hue=self.df_idx[response], s=60, markers='o',
                                              alpha=0.1, palette=palette_, edgecolor="black", legend=False)

                scatterplot.set_xlabel(Ax)
                scatterplot.set_ylabel(Ay)
                scatterplot.set_title(title)
                plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )

                if save:
                    plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

        elif plot_type.lower() == 'jitters':
            for i, calc_real in enumerate(self.calc_real):
                mds1_vec = np.transpose(calc_real[0, :])
                mds2_vec = np.transpose(calc_real[1, :])

                scatterplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, hue=self.df_idx[response], s=60, markers='o',
                                              alpha=0.1, palette=palette_, edgecolor="black", legend=False)

            if annotate:
                for index, label in enumerate(range(1, len(mds1_vec) + 1)):
                    plt.annotate(label, (mds1_vec[index] + x_off, mds2_vec[index] + y_off), size=8, style='italic')

            scatterplot.set_xlabel(Ax)
            scatterplot.set_ylabel(Ay)
            scatterplot.set_title(title)
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )

            if save:
                plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

        elif plot_type.lower() == 'uncertainty':
            for i, calc_real in enumerate(self.calc_real):
                mds1_vec = np.transpose(calc_real[0, :])
                mds2_vec = np.transpose(calc_real[1, :])
                if i == 0:
                    scatterplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, s=60, markers='o', alpha=0.3,
                                                  edgecolor="black", linewidths=2, palette=palette_,
                                                  hue=self.df_idx[response])
                else:
                    scatterplot = sns.scatterplot(x=mds1_vec, y=mds2_vec, s=60, markers='o', palette=palette_,
                                                  alpha=0.1,
                                                  legend=False, hue=self.df_idx[response])

            array_exp = np.mean(self.calc_real, axis=0)

            sns.scatterplot(x=array_exp[0, :], y=array_exp[1, :], s=50, marker='x', linewidths=4,
                            alpha=1, color='k', edgecolor="black", label='expectation',
                            legend=True)

            if annotate:
                for index, label in enumerate(range(1, len(array_exp[0, :]) + 1)):
                    plt.annotate(label, (array_exp[0, :][index] + x_off, array_exp[1, :][index] + y_off), size=8,
                                 style='italic')

            #  Aesthetics
            scatterplot.set_xlabel(Ax)
            scatterplot.set_ylabel(Ay)
            scatterplot.set_title(title)
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )
            plt.legend()

            if save:
                plt.savefig('2D registration jitters uncertainty w.r.t expectation for all realizations.tiff',
                            dpi=300, bbox_inches='tight')

            plt.show()

        else:
            print("Use a plot_type of `variation`, `jitters`, or `uncertainty`")


    def expectation(self, r_idx, Ax, Ay, verbose=False):
        """
        Calculates the expectation of all the calc_real.

        Arguments
        ---------
        r_idx : int
            Base case realization index.
        Ax : str
            Label for the x-axis.
        Ay : str
            Label for the y-axis.
        verbose : bool, optional
            Flag indicating whether to print verbose output. Default is False.

        Returns
        -------
        E : numpy.ndarray
            The expectation array.
        """

        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")

        base_real = self.all_real[r_idx]
        sig_x = np.var(base_real[:, 0])
        sig_y = np.var(base_real[:, 1])
        sig_eff = sig_x + sig_y

        E = np.mean(self.calc_real, axis=0)
        sigma_x = np.var(E[0, :])
        sigma_y = np.var(E[1, :])
        sigma_eff = sigma_x + sigma_y

        if verbose:
            print("The effective variance of the base case is", round(sig_eff, 4), "with a " + Ax + " variance of",
                  round(sig_x, 4), "and " + Ay + " variance of", round(sig_y, 4))
            print("The effective variance of the expected stabilized solution is", round(sigma_eff, 4),
                  "with a " + Ax + " variance of", round(sigma_x, 4), "and " + Ay + " variance of", round(sigma_y, 4))

        #  Update
        self.array_exp = E
        return E


    def expect_plotter(self, r_idx, Lx, Ly, xmin, xmax, ymin, ymax, save=True):
        """
        Plots the distributions of NDR (MDS) projections for the base case and stabilized expectation.

        Arguments
        ---------
        r_idx : int
            Base case realization index.
        Lx : str
            Label for the x-direction projection.
        Ly : str
            Label for the y-direction projection.
        xmin : float
            Minimum value for the x-axis.
        xmax : float
            Maximum value for the x-axis.
        ymin : float
            Minimum value for the y-axis.
        ymax : float
            Maximum value for the y-axis.
        save : bool, optional
            Flag indicating whether to save the plot. Default is True.

        Returns
        -------
        None
        """

        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")

        base_case_Lx = self.all_real[r_idx][:, 0]
        base_case_Ly = self.all_real[r_idx][:, 1]
        stabilized_expectation_Lx = self.array_exp[0, :]
        stabilized_expectation_Ly = self.array_exp[1, :]

        #  Plotting
        sns.kdeplot(base_case_Lx, label='Base case ' + Lx, color='blue')
        sns.kdeplot(stabilized_expectation_Lx, label=Lx + ' stabilized expectation', color='magenta', alpha=0.4)
        sns.kdeplot(base_case_Ly, label='Base case ' + Ly, color='green')
        sns.kdeplot(stabilized_expectation_Ly, label=Ly + ' stabilized expectation', color='orange', alpha=0.4)

        #  Aesthetics
        plt.legend(loc="best", fontsize=14)
        plt.xlabel('Projections', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.subplots_adjust(left=0.0, bottom=0.5, right=1.2, top=2.0, wspace=0.25, hspace=0.3)

        if save:
            plt.savefig('Comparisons for projections between stabilized solutions and base case distributions.tiff',
                        dpi=300, bbox_inches='tight')
        plt.show()


    def compare_plot(self, response, r_idx, Ax, Ay, x_off, y_off, cmap, annotate=True, save=True):
        """
        Plots a comparison between the base case realization and the ensemble expectation of stabilized solutions.

        Arguments
        ---------
        response : str
            Name of the response variable.
        r_idx : int
            Base case realization index.
        Ax : str
            Label for the x-axis.
        Ay : str
            Label for the y-axis.
        x_off : float
            Offset value for x-coordinate annotations.
        y_off : float
            Offset value for y-coordinate annotations.
        cmap : str or colormap object
            Colormap for the scatter plot.
        annotate : bool, optional
            Flag indicating whether to annotate the data points. Default is True.
        save : bool, optional
            Flag indicating whether to save the plot. Default is True.

        Returns
        -------
        None
        """

        if self.all_real is None:
            raise TypeError("Run run_rigid_MDS first.")
        if self.array_exp is None:
            raise TypeError("Run expectation first.")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        def plot_scatter(ax, x, y, hue, title):
            scatterplot = sns.scatterplot(x=x, y=y, hue=hue, s=60, markers='o', palette=cmap, edgecolor="black", ax=ax)

            if annotate:
                for i, txt in enumerate(self.df_idx[self.idx]):
                    scatterplot.annotate(txt, (x[i] + x_off, y[i] + y_off), size=10, style='italic')

            scatterplot.set_xlabel(Ax)
            scatterplot.set_ylabel(Ay)
            scatterplot.set_title(title)
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1., wspace=0.2, hspace=0.3, )

        plot_scatter(axs[0], self.all_real[r_idx][:, 0], self.all_real[r_idx][:, 1],
                     self.df_idx[response], "Base case realization at seed " + str(self.random_seeds[r_idx]))

        plot_scatter(axs[1], self.array_exp[0, :], self.array_exp[1, :],
                     self.df_idx[response],
                     "Expectation of Stabilized Solutions over " + str(self.num_realizations) + " realizations")

        if save:
            plt.savefig('Stabilized independent result vs expectation of stabilized results.tiff', dpi=300,
                        bbox_inches='tight')
        plt.show()


    def visual_model_check(self, norm_type, fig_name, array, expectation_compute=True, save=True):
        """
        Visualizes the model check for the projected data by computing the pairwise distances in the original and
        projected spaces for any scenario.

        Arguments
        ---------
        norm_type : str
            The type of distance norm to use. Valid values are 'L2' and 'L1'.
        fig_name : str
            The name of the figure file to save.
        array : ndarray
            The array containing the projected data.
        expectation_compute : bool, optional
            Flag indicating whether to compute the expectation of the stabilized solution. Default is True.
        save : bool, optional
            Flag indicating whether to save the plot. Default is True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If invalid norm_type is used
        """

        stabilized_expected_proj = np.transpose(array[:2, :]) if expectation_compute else array.copy()

        #  Distance calculation based on norm_type
        if norm_type.upper() == 'L2':
            # dists = manhattan_distances(self.df).ravel()
            # nonzero = dists != 0
            # dists = dists[nonzero]
            # projected_dists = manhattan_distances(stabilized_expected_proj).ravel()[nonzero]
            dists = pairwise_distances(self.df, metric='manhattan').ravel()
            projected_dists = pairwise_distances(stabilized_expected_proj, metric='manhattan').ravel()
        elif norm_type.upper() == 'L1':
            # dists = euclidean_distances(self.df, squared=False).ravel()
            # nonzero = dists != 0
            # dists = dists[nonzero]
            # projected_dists = euclidean_distances(stabilized_expected_proj, squared=False).ravel()[nonzero]
            dists = pairwise_distances(self.df, metric='euclidean').ravel()
            projected_dists = pairwise_distances(stabilized_expected_proj, metric='euclidean').ravel()
        else:
            raise ValueError("Invalid norm_type. Valid values are 'L2' and 'L1'.")

        #  Filter out zero distances
        nonzero_indices = dists != 0
        dists = dists[nonzero_indices]
        projected_dists = projected_dists[nonzero_indices]

        #  Calculate distance ratios
        rates = projected_dists / dists
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)

        #  Make plot
        plt.subplot(221)
        plt.scatter(dists, projected_dists, c='red', alpha=0.2, edgecolor='black')
        plt.arrow(0, 0, 200, 200, width=0.02, color='black', head_length=0.0, head_width=0.0)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.xlabel("Pairwise Distance: original space")
        plt.ylabel("Pairwise Distance: projected space")
        plt.title("Pairwise Distance: Projected to 2 components")

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

        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.7, top=2.3, wspace=0.2, hspace=0.3)

        if save:
            plt.savefig(fig_name + '.tiff', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Distance Ratio, mean: {mean_rate:.4f}, standard deviation: {std_rate:.4f}.")


    @staticmethod
    def convex_hull(array, title, x_off, y_off, Ax, Ay, num_OOSP=None, expectation_compute=True, make_figure=True, n_case=True,
                    annotate=True, save=True):
        """
        Computes the convex hull of the given array of points and visualizes it.

        Arguments
        ---------
        array : ndarray
            The array of points.
        num_OOSP : int, optional
            Number of OOSP samples added within the 95% confidence interval
        title : str
            The title of the plot.
        x_off : float
            The x-offset for annotation.
        y_off : float
            The y-offset for annotation.
        Ax : str
            The label for the x-axis.
        Ay : str
            The label for the y-axis.
        expectation_compute : bool, optional
            Flag indicating whether to compute the expectation of the stabilized solution. Default is True.
        make_figure : bool, optional
            Flag indicating whether to make the figure. Default is True.
        n_case : bool, optional
            Flag indicating whether it is an N-case scenario. Default is True.
        annotate : bool, optional
            Flag indicating whether to annotate the sample indices. Default is True.
        save : bool, optional
            Flag indicating whether to save the plot. Default is True.

        Returns
        -------
        tuple or str
            If the convex polygon assumption is met, returns a tuple containing the points, hull, and vertices.
            Otherwise, returns a string indicating that the convex polygon assumption is not met.
        """

        my_points = np.transpose(array[:2, :]) if expectation_compute else array[0][:, :2]
        hull = ConvexHull(my_points)
        vertices = my_points[hull.vertices]
        polygon = Polygon(vertices)

        #  if not polygon.is_convex:
        if is_convex_polygon(polygon) is False:
            return "Convex polygon assumption not met, do not use this workflow"

        #  Make plot
        if make_figure:
            if n_case or num_OOSP is None:
                #  For N-sample case
                plt.scatter(my_points[:, 0], my_points[:, 1], marker='o', s=50, color='white', label='sample', edgecolors="black")
            else:
                #  For OOSP included case
                plt.scatter(my_points[:-num_OOSP, 0], my_points[:-num_OOSP, 1], marker='o', s=50, color='white', label='sample', edgecolors="black")
                plt.scatter(my_points[-num_OOSP:, 0], my_points[-num_OOSP:, 1], marker='*', s=90, color='black', label='OOSP', edgecolors="black")

            if annotate:
                for index, label in enumerate(range(1, len(my_points) + 1)):
                    plt.annotate(label, (my_points[:, 0][index] + x_off, my_points[:, 1][index] + y_off), size=8,
                                 style='italic')

            for simplex in hull.simplices:
                plt.plot(my_points[simplex, 0], my_points[simplex, 1], 'r--')
                plt.fill(my_points[hull.vertices, 0], my_points[hull.vertices, 1], c='yellow', alpha=0.01)

            plt.title(title)
            plt.xlabel(Ax)
            plt.ylabel(Ay)
            plt.legend(loc="best", fontsize=12)

            plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )

            if save:
                plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')

            plt.show()
        return my_points, hull, vertices


    def marginal_dbn(self, save=True):
        """
        Computes the marginal probability density distributions of each predictor, along with shaded regions representing
        the standard deviation thresholds (+/-1, +/-2, +/-3 standard deviations from the mean) for each predictor.

        Arguments
        ---------
        save : bool, optional
            Flag indicating whether to save the plot. Default is True.

        Returns
        -------
        None
        """

        ns_features = self.df.columns.tolist()
        N = 10
        fig, axs = plt.subplots(1, len(ns_features), figsize=(6 * len(ns_features), 4))

        for feat, ax in zip(ns_features, axs):
            x = self.df[feat]
            stdev = np.std(x)
            mean = np.mean(x)
            sns.kdeplot(x, fill=True, ax=ax)

            for i in range(1, 4):
                x1 = np.linspace(mean - i * stdev, mean - (i - 1) * stdev, N)
                x2 = np.linspace(mean - (i - 1) * stdev, mean + (i - 1) * stdev, N)
                x3 = np.linspace(mean + (i - 1) * stdev, mean + i * stdev, N)
                x = np.concatenate((x1, x2, x3))
                x[(mean - (i - 1) * stdev < x) & (x < mean + (i - 1) * stdev)] = np.nan
                y = norm.pdf(x, mean, stdev)
                ax.fill_between(x, y, alpha=0.5)

            #  Aesthetics
            ax.set_xlabel('Normal scores for ' + feat[3:])
            ax.set_xticks(ticks=np.arange(-5, 5))
            ax.set_ylabel("Marginal probability density")

        std_patches = [
            mpatches.Patch(color='sandybrown', label='+/- 1' r'$\sigma$'),
            mpatches.Patch(color='darkseagreen', label='+/- 2' r'$\sigma$'),
            mpatches.Patch(color='indianred', label='+/- 3' r'$\sigma$')
        ]
        plt.legend(handles=std_patches)
        plt.subplots_adjust(wspace=0.3)

        if save:
            plt.savefig('Marginal_distributions.tiff', dpi=300, bbox_inches='tight')

        plt.show()
        return


# noinspection PyUnboundLocalVariable
class RigidTransf_NPlus(RigidTransformation):
    def __init__(self, df, features, idx, num_OOSP, num_realizations, base_seed, start_seed, stop_seed,
                 dissimilarity_metric, dim_projection):
        super().__init__(df, features, idx, num_OOSP, num_realizations, base_seed, start_seed, stop_seed,
                         dissimilarity_metric, dim_projection)
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
        """
        Stabilizes anchor points between two arrays using convex hulls and rigid transformations.

        Arguments
        ---------
        array1 : numpy.ndarray
            The anchor points for the N-sample case.
        array2 : numpy.ndarray
            The anchor points for the N+1-sample case.
        hull_1 : scipy.spatial.qhull.ConvexHull
            The convex hull for array1.
        hull_2 : scipy.spatial.qhull.ConvexHull
            The convex hull for array2.
        normalize_projections : bool, optional
            Indicates whether to normalize the stabilized anchor points. Default is True.

        Returns
        -------
        anchors1 : numpy.ndarray
            The stabilized anchor points for the N-sample case.
        anchors2 : numpy.ndarray
            The stabilized anchor points for the N+1-sample case.
        R_anchors : numpy.ndarray
            The rotation matrix for the stabilized anchor points.
        t_anchors : numpy.ndarray
            The translation matrix for the stabilized anchor points.
        rmse_err_anchors : float
            The root mean squared error between the stabilized anchor points and array1.
        stable_coords_anchors : numpy.ndarray
            The normalized stabilized anchor points.
        stable_coords_alldata : numpy.ndarray
            The stabilized representation of all data points in the N+1-sample case.
        rmse_err_alldata : float
            The root mean squared error between the stabilized representation and array2.

        Raises
        ------
        TypeError: If dim_projection is not '2D' or '3D'.
        """

        # Obtain the anchor points for n and n+1 scenarios
        vertices_index = hull_1.vertices
        vertices2_index = hull_2.vertices

        # Find the common anchor points/vertices between anchors in N and N+1 sample case
        common_vertices_index = np.intersect1d(vertices_index, vertices2_index)
        common_vertices2_index = np.intersect1d(vertices2_index, vertices_index)

        # Access the corresponding anchor points using the common indexes
        case1_anchors = array1[common_vertices_index]
        case2_anchors = array2[common_vertices2_index]

        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            anchors1 = case1_anchors[:, :2]
            anchors2 = case2_anchors[:, :2]
        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            anchors1 = np.column_stack((case1_anchors[:, :2], np.zeros(len(case1_anchors))))
            anchors2 = np.column_stack((case2_anchors[:, :2], np.zeros(len(case2_anchors))))
        else:
            raise TypeError("Use an LDS projection of '2D' or '3D' as dim_projection variable input in class.")

        # Recover the rotation and translation matrices R,t, respectively for the stable anchor points in n+1 to
        # match anchors in the n-case scenario
        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            R_anchors, t_anchors = rigid_transform_2D(np.transpose(anchors2), np.transpose(anchors1))
            # Compare the recovered R and t with the original by creating a new coordinate scheme via prior solutions
            # of R, t
            new_coord_anchors = (R_anchors @ np.transpose(anchors2)) + np.expand_dims(t_anchors, axis=1)

        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            R_anchors, t_anchors = rigid_transform_3D(np.transpose(anchors2), np.transpose(anchors1))
            # Compare the recovered R and t with the original by creating a new coordinate scheme via prior solutions
            # of R, t
            new_coord_anchors = (R_anchors @ np.transpose(anchors2)) + t_anchors

        # Find the rmse as an error check between estimated anchor points in n+1 scenario and anchor points in n-scenario
        rmse_err_anchors = rmse(new_coord_anchors, anchors1)

        # Create a convex hull polygon of the normalized stabilized anchor points
        stable_coords_anchors = np.transpose(new_coord_anchors[:2, :])

        if normalize_projections:
            scaler = StandardScaler()
            stable_coords_anchors = scaler.fit_transform(stable_coords_anchors)

        if self.dim_projection == '2D':  # i.e., if LDS is 2D
            anchors1 = case1_anchors[:, :2]
            anchors2 = case2_anchors[:, :2]
            stable_anchors_array = array2[:, :2]

            # Use the R and t matrix from the stabilized anchor solution and apply it to all samples in the n+1 scenario
            # to obtain the now stabilized solution for every sample point.
            new_coords_alldata = R_anchors @ np.transpose(stable_anchors_array) + np.expand_dims(t_anchors, axis=1)

            # # Computationally heavier method, better to use above anchor registration method as proposed. To be used
            # when there is no SVD rigid transformation possible due to deformation of points and OOSP from the tails.
            # stable_anchors_array = np.column_stack((array2[:len(array2) - self.num_OOSP, 0], array2[:len(array2) - self.num_OOSP, 1]))
            # R_all, t_all = rigid_transform_2D(np.transpose(stable_anchors_array), np.transpose(array1))
            # new_coords_alldata = (R_all @ np.transpose(array2)) + np.expand_dims(t_all, axis=1)

        elif self.dim_projection == '3D':  # i.e., if LDS is 3D
            anchors1 = np.column_stack((case1_anchors[:, :2], np.zeros(len(case1_anchors))))
            anchors2 = np.column_stack((case2_anchors[:, :2], np.zeros(len(case2_anchors))))
            stable_anchors_array = np.column_stack((array2[:, :2], np.zeros(len(case2_anchors))))

            # Use the R and t matrix from the stabilized anchor solution and apply it to all samples in the n+1 scenario
            # to obtain the now stabilized solution for every sample point.
            new_coords_alldata = (R_anchors @ np.transpose(stable_anchors_array)) + t_anchors

            # # Computationally heavier method, better to use above anchor registration method as proposed. To be used
            # when there is no SVD rigid transformation possible due to deformation of points and OOSP from the tails
            # stable_anchors_array = np.column_stack(
            #     (array2[:len(array2) - self.num_OOSP, 0], array2[:len(array2) - self.num_OOSP, 1], [0] * (len(array2) - self.num_OOSP)))
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
        return anchors1, anchors2, R_anchors, t_anchors, rmse_err_anchors, stable_coords_anchors, stable_coords_alldata, rmse_err_alldata


    def stable_anchor_visuals(self, Ax, Ay, x_off, y_off, annotate=True, save=True):
        """
        Visualizes the base case anchors, the N+1 case anchors, and the stabilized anchor solution.

        Arguments
        ---------
        Ax : str
            Label for the x-axis.
        Ay : str
            Label for the y-axis.
        x_off : float
            Offset for the x-coordinate of the annotations.
        y_off : float
            Offset for the y-coordinate of the annotations.
        annotate : bool, optional
            Flag indicating whether to annotate the anchor points with labels. Defaults to True.
        save : bool, optional
            Flag indicating whether to save the plot as an image. Defaults to True.

        Returns
        -------
        None
        """

        #  Make plot
        fig, axes = plt.subplots(1, 3)

        anchor_labels = self.common_vertices_index if annotate else None

        # Plot anchors from N sample case
        axes[0].scatter(self.anchors1[:, 0], self.anchors1[:, 1], marker='o', s=50, color='blue', edgecolors="black")
        axes[0].set_aspect('auto')
        axes[0].set_title('Anchors from N sample case', size=14)
        axes[0].set_xlabel(Ax, size=14)
        axes[0].set_ylabel(Ay, size=14)
        axes[0].tick_params(axis='both', which='major', labelsize=12)
        if annotate:
            for index, label in enumerate(anchor_labels):
                axes[0].annotate(label, (self.anchors1[:, 0][index] + x_off, self.anchors1[:, 1][index] + y_off),
                                 size=10, style='italic')

        # Plot anchors from N+1 sample case
        axes[1].scatter(self.anchors2[:, 0], self.anchors2[:, 1], marker='o', s=50, color='blue', edgecolors="black")
        axes[1].set_aspect('auto')
        axes[1].set_title('Anchors from N+1 sample case', size=14)
        axes[1].set_xlabel(Ax, size=14)
        axes[1].set_ylabel(Ay, size=14)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        if annotate:
            for index, label in enumerate(self.common_vertices2_index):
                axes[1].annotate(label, (self.anchors2[:, 0][index] + x_off, self.anchors2[:, 1][index] + y_off),
                                 size=10, style='italic')

        # Plot stabilized anchor solution
        axes[2].scatter(self.stable_coords_anchors[:, 0], self.stable_coords_anchors[:, 1], marker='o', s=50,
                        color='blue', edgecolors="black")
        axes[2].set_aspect('auto')
        axes[2].set_title('Stabilized anchor solution', size=14)
        axes[2].set_xlabel(Ax, size=14)
        axes[2].set_ylabel(Ay, size=14)
        axes[2].tick_params(axis='both', which='major', labelsize=12)
        if annotate:
            for index, label in enumerate(anchor_labels):
                axes[2].annotate(label, (self.stable_coords_anchors[:, 0][index] + x_off,
                                         self.stable_coords_anchors[:, 1][index] + y_off), size=10, style='italic')

        plt.subplots_adjust(left=0.0, bottom=0.0, right=3., top=1.3, wspace=0.25, hspace=0.3)
        if save:
            plt.savefig('Anchor sets & Stabilized Anchor set Solution.tiff', dpi=300, bbox_inches='tight')

        plt.show()


    def stable_representation(self, title, Ax, Ay, x_off, y_off, annotate=True, save=True):
        """
        Visualizes the n+1 case for all samples with a stabilized representation obtained in the n-case.
        The visualization seen is invariant to rotation, translation, and reflection transformations.

        Arguments
        ---------
        title : str
            The title of the plot.
        Ax : str
            The label for the x-axis.
        Ay : str
            The label for the y-axis.
        x_off : float
            The offset value to adjust the x-coordinate of the annotations.
        y_off : float
            The offset value to adjust the y-coordinate of the annotations.
        annotate : bool, optional
            Indicates whether to annotate the data points. Defaults to True.
        save : bool, optional
            Indicates whether to save the plot as an image file. Defaults to True.

        Returns
        -------
        None
        """

        # Update
        self.Ax = Ax
        self.Ay = Ay
        self.x_off = x_off
        self.y_off = y_off
        self.title = title

        # Make plot
        fig, ax = plt.subplots()
        ax.scatter(self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 0], self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 1],
                    marker='o', label='sample', s=50, color='white', edgecolors="black")
        ax.scatter(self.stable_coords_alldata[(len(self.stable_coords_alldata) - self.num_OOSP):, 0], self.stable_coords_alldata[(len(self.stable_coords_alldata) - self.num_OOSP):, 1],
                   marker='*', label='OOSP', color='k', s=90)

        if annotate:
            for label, x, y in zip(range(1, len(self.stable_coords_alldata) + 1),
                                   self.stable_coords_alldata[:, 0] + x_off,
                                   self.stable_coords_alldata[:, 1] + y_off):
                ax.annotate(label, (x, y), size=8, style='italic')

        # Aesthetics
        ax.set_title(title)
        ax.set_xlabel(Ax)
        ax.set_ylabel(Ay)
        plt.legend(loc="best", fontsize=12)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )

        if save:
            plt.savefig('Stabilized N+1 case with same representation as N case.tiff', dpi=300, bbox_inches='tight')

        plt.show()


    def stabilized_all_plotter(self, dataframe, hue_, palette_, annotate=True, n_case=True, save=True):
        if hue_ is not None:
            cmap = "rocket_r" if palette_ == 1 else "bright"
            categories = dataframe[hue_].unique()
            num_categories = len(categories)

            # Define the color palette
            palette = sns.color_palette(cmap, n_colors=num_categories)
            category_to_color = dict(zip(categories, palette))

            scatter_colors = [category_to_color[category] for category in dataframe[hue_]]

        if n_case:
            plt.scatter(self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 0],
                        self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 1],
                        marker='o', label='sample',
                        s=50, linewidths=0.5,
                        c=scatter_colors[:len(self.stable_coords_alldata) - self.num_OOSP],
                        edgecolors="black")

            if annotate:
                for label, x, y in zip(range(1, len(self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 0]) + 1),
                                       self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 0] + self.x_off,
                                       self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 1] + self.y_off):
                    plt.annotate(label, (x, y), size=8, style='italic')

            # Aesthetics
            plt.title(self.title)
            plt.xlabel(self.Ax)
            plt.ylabel(self.Ay)


        else:
            plt.scatter(self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 0],
                        self.stable_coords_alldata[:len(self.stable_coords_alldata) - self.num_OOSP, 1],
                        marker='o', label='sample',
                        s=50, linewidths=0.5,
                        c=scatter_colors[:len(self.stable_coords_alldata) - self.num_OOSP],
                        edgecolors="black")

            plt.scatter(self.stable_coords_alldata[(len(self.stable_coords_alldata) - self.num_OOSP):, 0],
                        self.stable_coords_alldata[(len(self.stable_coords_alldata) - self.num_OOSP):, 1],
                        marker='*', label='OOSP',
                        s=200, linewidths=0.5,
                        c=scatter_colors[(len(self.stable_coords_alldata) - self.num_OOSP):],
                        edgecolors="black")

            if annotate:
                for label, x, y in zip(range(1, len(self.stable_coords_alldata) + 1),
                                       self.stable_coords_alldata[:, 0] + self.x_off,
                                       self.stable_coords_alldata[:, 1] + self.y_off):
                    plt.annotate(label, (x, y), size=8, style='italic')

            # Aesthetics
            plt.title(self.title)
            plt.xlabel(self.Ax)
            plt.ylabel(self.Ay)

        # Add custom colorbar
        if hue_ is not None:
            unique_colors = [category_to_color[category] for category in categories]
            cmap = ListedColormap(unique_colors)
            bounds =  range(num_categories + 1)
            tick_positions = [i + 0.5 for i in bounds[:-1]]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ticks=tick_positions,
                                    boundaries=bounds, spacing='proportional')
            colorbar.set_ticklabels(categories)
            colorbar.set_label(hue_, rotation=270, labelpad=30, size=12)

        plt.legend(loc="best", fontsize=12)
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1., top=1.3, wspace=0.3, hspace=0.3, )

        if save:
            plt.savefig(self.title + '.tiff', dpi=300, bbox_inches='tight')
        plt.show()
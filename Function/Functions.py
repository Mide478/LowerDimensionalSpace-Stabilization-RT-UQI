import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS          # multidimensional scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull

def Normalizer(df_original, feats):
    """

    This function normalizes the dataframe of choice between [0.01,1]

    Arguments
    ---------
    df: a dataframe consisting of features to be normalized

    feats: a list consisting of features column names to be normalized

    """
    df = df_original.copy()
    x = df.loc[:, feats].values
    scaler = MinMaxScaler(feature_range=(0.01, 1))
    xs = scaler.fit_transform(x)

    ns_feats = []
    for i in range(0, len(feats)):
        df['NS_' + feats[i]] = xs[:, i]
        ns_feats.append('NS_' + feats[i])

    return df


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


def generate_random_seeds(seed, num_realizations, lower_bound, upper_bound):
    """
    :param seed: TODO
    :param num_realizations: TODO
    :param lower_bound: TODO
    :param upper_bound: TODO
    :return: TODO
    """
    random_seeds = []
    random.seed(seed)              # random number seed is set to ensure reproducibility of same realization seeds,
    # which is different for every iteration in the realization
    for i in range(num_realizations):
        random_value = random.randint(lower_bound, upper_bound)
        random_seeds.append(random_value)
    return random_seeds


def run_rigid_MDS(df, ns_features, num_realizations, base_seed, start_seed, stop_seed):

    """
    :param df: TODO
    :param ns_features: TODO
    :param num_realizations: maximum number of random samples
    :param base_seed: TODO
    :param start_seed: TODO
    :param stop_seed: TODO
    :return: TODO
    """

    # Arrays below store random values for every parameter changing using the utility functions defined later in the
    # code for each realization
    random_seeds = generate_random_seeds(base_seed, num_realizations, start_seed, stop_seed)

    mds1 = [] # MDS projection 1
    mds2 = [] # MDS projection 2
    norm_stress = []
    all_real = [] # All realizations prepared for rigid transform
    t = []
    r = []
    all_rmse = []
    calc_real = [] # analytical estimation of each realization from R,T recovered

    for i in range(0, num_realizations):
        embedding_subset = MDS(n_components=2,n_init = 20,max_iter = 1000,random_state = random_seeds[i])
        mds_transformed_subset = embedding_subset.fit_transform(df[ns_features])
        raw_stress = embedding_subset.stress_
        dissimilarity_matrix = embedding_subset.dissimilarity_matrix_
        stress_1 = np.sqrt(raw_stress / (0.5 * np.sum(dissimilarity_matrix**2)))
        norm_stress.append(stress_1) # [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]
        mds1.append(mds_transformed_subset[:,0])
        mds2.append(mds_transformed_subset[:,1])
        real_i = np.column_stack((mds1[i],mds2[i],[0]*len(mds1[i]))) # stack projections for all realizations
        all_real.append(real_i)

    # Make the LD space invariant to  translation, rotation, reflection/flipping, This applies the proposed method to
    # all realization and the base case individually to yield a unique solution.
    for i in range(1,len(all_real)):
        # Recover the rotation and translation matrices, R,T respectively for each realization
        ret_R, ret_T = rigid_transform_3D(np.transpose(all_real[i]), np.transpose(all_real[0]))
        t.append(ret_T)
        r.append(ret_R)

        # Compare the recovered R and T with the base case by creating a new coordinate scheme via prior
        # solutions of r, and t
        new_coord = (ret_R@np.transpose(all_real[i])) + ret_T
        calc_real.append(new_coord)

        # Find the rmse as an error check between corrected realization and base case
        rmse_err = rmse(new_coord, all_real[0])
        all_rmse.append(rmse_err)
    return random_seeds, all_real, calc_real, all_rmse, norm_stress


def rmse(array1, array2):
    """

    :param array1: recovered realization "i" array from R, T calculation
    :param array2: base case
    :return: TODO
    """

    var1 = np.transpose(array1) - array2
    var1 = var1 * var1
    var1 = np.sum(var1)
    rmse = np.sqrt(var1 / len(array1[0, :]))
    return rmse


def real_plotter(df, idx, response, array1, r_idx, random_seeds, Ax, Ay, title, x_off, y_off, cmap, array2=None):
    """
    :param df: TODO
    :param idx: TODO
    :param response: TODO
    :param array1: TODO - all_real
    :param r_idx: TODO - calc_real
    :param random_seeds: TODO
    :param Ax: TODO
    :param Ay: TODO
    :param title: TODO
    :param x_off: TODO
    :param y_off: TODO
    :param cmap: TODO
    :param array2: TODO
    :return: TODO
    """

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
            pairplot = sns.scatterplot(x=array1[i][:, 0], y=array1[i][:, 1], hue=df[response], s=60, markers='o',
                                       palette=cmap, edgecolor="black", ax=ax)
            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title( title[i] + str(r_idx[i]) + " at seed " + str(random_seeds[i]))
            for j, txt in enumerate(df[idx]):
                pairplot.annotate(txt, (array1[i][:, 0][j]+x_off, array1[i][:, 1][j]+y_off), size=10, style='italic')

    else:
        for k in range(1, len(r_idx)):
            ax = plt.subplot(num_rows, num_cols, k)
            pairplot = sns.scatterplot(x=array2[k][0], y=array2[k][1], hue=df[response], s=60, markers='o',
                                       palette=cmap, edgecolor="black", ax=ax)
            pairplot.set_xlabel(Ax)
            pairplot.set_ylabel(Ay)
            pairplot.set_title("Stabilized solution for " + title[k].lower() + str(r_idx[k]) + " at seed " +
                               str(random_seeds[k-1]))
            for l, txt in enumerate(df[idx]):
                pairplot.annotate(txt, (array2[k][0][l]+x_off, array2[k][1][l]+y_off),size=10, style='italic')

        # Add base case to subplot for direct comparison of stabilized solution obtained
        ax = plt.subplot(num_rows, num_cols, k+1)
        pairplot = sns.scatterplot(x=array1[0][:, 0], y=array1[0][:, 1],hue = df[response], s=60, markers='o',
                                   palette=cmap, edgecolor="black", ax=ax)
        pairplot.set_xlabel(Ax)
        pairplot.set_ylabel(Ay)
        pairplot.set_title(title[0] + str(r_idx[0]) + " at seed " + str(random_seeds[0]))

    # Figure info
    ax.set_aspect('auto')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Aesthetics
    plt.subplots_adjust(left=0.0, bottom=0.0, right=2., top=2., wspace=0.3, hspace=0.3,)
    plt.savefig( 'Variations with seeds 2x2 for data subset with tracking.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


def expectation(array1, array2, r_idx, Ax, Ay, verbose=False):
    """
    :param array1: TODO - all_real
    :param array2: TODO - calc_real
    :param r_idx: TODO - base case realization index
    :param Ax: TODO
    :param Ay: TODO
    :param verbose: TODO
    :return: TODO - expectation of all the calc_real
    """

    # Base case
    sig_x = np.var(array1[r_idx][:, 0])
    sig_y = np.var(array1[r_idx][:, 1])
    sig_eff = sig_x + sig_y

    # Expectation of stabilized solution over all realizations for each sample
    E = np.mean(array2, axis=0)
    sigma_x = np.var(E[0, :])
    sigma_y = np.var(E[1, :])
    sigma_eff = sigma_x + sigma_y

    if verbose:
        print("The effective variance of the base case is", round(sig_eff, 4), "with a " + Ax + " variance of",
              round(sig_x, 4), " and " + Ay + " variance of", round(sig_y, 4))
        print("\nThe effective variance of the expected stabilized solution is", round(sigma_eff,4), "with a " + Ax +
              " variance of", round(sigma_x, 4), " and " + Ay + " variance of", round(sigma_y, 4))
    return E


def E_plotter(array1, array_exp, r_idx, Lx, Ly, xmin, xmax, ymin, ymax):
   """
   :param array1: TODO - all_real
   :param array_exp: TODO - expectation array
   :param r_idx: TODO - base case index
   :param Lx: TODO - x-axis label
   :param Ly: TODO - y-axis label
   :param xmin: TODO
   :param xmax: TODO
   :param ymin: TODO
   :param ymax: TODO
   :return: TODO
   """

    # For x-direction projection in LD space
   sns.kdeplot(array1[r_idx][:, 0], label='Base case ' + Lx, color='blue')
   sns.kdeplot(array_exp[0, :], label=Lx + ' stabilized expectation', color='magenta', alpha=0.4)

    # For y-direction projection in LD space
   sns.kdeplot(array1[r_idx][:, 1], label= 'Base case ' + Ly, color= 'green')
   sns.kdeplot(array_exp[1, :], label=Ly + ' stabilized expectation', color='orange', alpha=0.4)

    # Aesthetics
   plt.legend(loc="best", fontsize=14)
   plt.xlabel('Projections', fontsize=14)
   plt.ylabel('Density', fontsize=14)
   plt.xlim(xmin,xmax)
   plt.ylim(ymin,ymax)
   plt.tick_params(axis='both', which='major', labelsize=12)
   plt.subplots_adjust(left=0.0, bottom=0.5, right=1.2, top=2.0, wspace=0.25, hspace=0.3)
   plt.savefig('Comparisons for projections between stabilized solutions and base case distributions.tiff', dpi=300,
               bbox_inches='tight')
   plt.show()
   return


def compare_plot(df, idx, response, array1, r_idx, num_realizations, array_exp, random_seeds, Ax, Ay, x_off, y_off,
                 cmap):
    """
    :param df: TODO
    :param idx: TODO
    :param response: TODO
    :param array1: TODO
    :param r_idx: TODO
    :param num_realizations: TODO
    :param array_exp: TODO
    :param random_seeds: TODO
    :param Ax: TODO
    :param Ay: TODO
    :param x_off: TODO
    :param y_off: TODO
    :param cmap: TODO
    :return: TODO
    """

    plt.subplot(121)
    pairplot = sns.scatterplot(x=array1[r_idx][:, 0], y=array1[r_idx][:, 1], hue=df[response], s=60, markers='o',
                               palette=cmap, edgecolor="black")
    for i, txt in enumerate(df[idx]):
        pairplot.annotate(txt, (array1[r_idx][:,0][i]+x_off, array1[r_idx][:,1][i]+y_off),size=10, style='italic')
    pairplot.set_xlabel(Ax)
    pairplot.set_ylabel(Ay)
    pairplot.set_title("Base case realization at seed " + str(random_seeds[r_idx]))

    plt.subplot(122)
    pairplot = sns.scatterplot(x=array_exp[0, :], y=array_exp[1, :], hue=df[response], s=60, markers='o', palette=cmap,
                               edgecolor="black")
    for i, txt in enumerate(df[idx]):
        pairplot.annotate(txt, (array_exp[0, :][i]+x_off, array_exp[1, :][i]+y_off), size=10, style='italic')
    pairplot.set_xlabel(Ax)
    pairplot.set_ylabel(Ay)
    pairplot.set_title("Expectation of Stabilized Solutions over " + str(num_realizations) + " realizations") # subtract
    # num_realizations from 1 since base case is a realization too?!

    plt.subplots_adjust(left=0.0, bottom=0.0, right=2., top=1., wspace=0.3, hspace=0.3,)
    plt.savefig( 'Stabilized independent result vs expectation of stabilized results.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


# check if data is a convex polygon
def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

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



def matrix_scatter(dataframe, feat_title, left_adj, bottom_adj, right_adj, top_adj, wspace, hspace, title, palette_,
                   hue_=None):
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
    """

    # Hue assignment
    if hue_ is not None:
        hue_ = hue_

        # Palette assignment
        if palette_ == 1:
            palette_ = sns.color_palette("rocket_r", n_colors=len(dataframe[hue_].unique()))

        elif palette_ == 2:
            palette_ = sns.color_palette("bright", n_colors=len(dataframe[hue_].unique()))
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

    sns.pairplot(dataframe, vars=feat_title, markers='o', diag_kws={'edgecolor':'black'},
                 plot_kws=dict(s=50, edgecolor="black", linewidth=0.5),hue=hue_, corner=True,
                 palette=palette_)
    plt.subplots_adjust(left=left_adj, bottom=bottom_adj, right=right_adj, top=top_adj, wspace=wspace, hspace=hspace)
    plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return


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
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    xs = StandardScaler().fit_transform(x)

    ns_feats = []
    for i, feature in enumerate(features):
        df['NS_' + feature] = xs[:, i]
        ns_feats.append('NS_' + feature)

    if keep_only_std_features:
        df = df.loc[:, ns_feats]

    return df


def bivariate_plotter(array, palette_, response, title, plot_type, dataframe, Ax, Ay):
    """

    :param array:
    :param palette_:
    :param response:
    :param title:
    :param plot_type:
    :param dataframe:
    :param Ax:
    :param Ay:
    :return:
    """

    df = dataframe.copy(deep=True)
    plot_type = plot_type.lower()

    # Palette assignment
    if palette_ == 1:
        palette_ = sns.color_palette("rocket_r", n_colors=len(np.unique(df[response].values))+1)

    elif palette_ == 2:
        palette_ = sns.color_palette("bright", n_colors=len(np.unique(df[response].values))+1)

    else:
        palette_ = None

    for i in range(0,len(array)):
        if plot_type == 'variation':
            mds1_vec = array[i][:,0]
            mds2_vec = array[i][:,1]

        elif plot_type == 'jitters':
            mds1_vec = np.transpose(array[i][0, :])
            mds2_vec = np.transpose(array[i][1, :])
        else:
            print("Use a plot_type of variation or jitters")

        pairplot=sns.scatterplot(x=mds1_vec, y=mds2_vec, hue=df[response], s=60, markers='o', alpha=0.1,
                        palette=palette_, edgecolor="black", legend=False)
        pairplot.set_xlabel(Ax)
        pairplot.set_ylabel(Ay)
        pairplot.set_title(title)
        plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')
    return

def visual_model_check(dataframe, features, fig_name, expectation_array):
    """

    :param dataframe:
    :param features:
    :param fig_name:
    :param expectation_array:
    :return:
    """

    # Obtain dataframe with the standardized predictor features
    df = dataframe[features]

    # Grab the expectation of the stabilized solution
    stabilized_expected_proj = np.transpose(expectation_array[:2, :])

    # insert distortion visual
    dists = euclidean_distances(df, squared=False).ravel()
    nonzero = dists != 0   # select only non-identical samples pairs
    dists = dists[nonzero]
    projected_dists = euclidean_distances(stabilized_expected_proj, squared=False).ravel()[nonzero]

    plt.subplot(221)
    plt.scatter(dists,projected_dists,c='red',alpha=0.2,edgecolor = 'black')
    plt.arrow(0,0,200,200,width=0.02,color='black',head_length=0.0,head_width=0.0)
    plt.xlim(0,15); plt.ylim(0,15)
    plt.xlabel("Pairwise Distance: original space")
    plt.ylabel("Pairwise Distance: projected space")
    plt.title("Pairwise Distance: Projected to 2 components")

    rates = projected_dists / dists
    print("Distance Ratio, mean: %0.4f, standard deviation %0.4f." % (np.mean(rates), np.std(rates)))

    plt.subplot(222)
    plt.hist(rates, bins=50, range=(0.5, 1.5),color = 'red', alpha = 0.2, edgecolor='k')
    plt.xlabel("Distance Ratio: projected / original")
    plt.ylabel("Frequency")
    plt.title("Pairwise Distance: Projected to 2 Components")

    plt.subplot(223)
    plt.hist(dists, bins=50, range=(0., 15.),color = 'red', alpha = 0.2, edgecolor='k')
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title("Pairwise Distance: Original Data")

    plt.subplot(224)
    plt.hist(projected_dists, bins=50, range=(0., 15.),color = 'red', alpha = 0.2, edgecolor='k')
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Frequency")
    plt.title("Pairwise Distance: Projected to 2 Components")

    # Aesthetics
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.7, top=2.3, wspace=0.2, hspace=0.3)
    plt.savefig(fig_name + '.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    return

def convex_hull(array, title, x_off, y_off, Ax, Ay):

    # Using samples from either n-case or n+1 case scenario to make a convex hull i.e., convex polygon
    my_points = array[0][:,:2] # all samples in projected space
    hull = ConvexHull(my_points)

    # Check for point in polygon
    vertices = my_points[hull.vertices] # the anchors as an array
    # plt.scatter(my_points[hull.vertices][:,0], my_points[hull.vertices][:,1]) # vertices only in normalized space
    polygon = Polygon(vertices)

    # check if data is a strict convex polygon
    binary_bool = is_convex_polygon(polygon)
    if binary_bool is False:
        return "Convex polygon assumption not met, do not use this workflow"

    else:
        # Make sample point visuals
        plt.scatter(my_points[:, 0], my_points[:, 1], marker='o', s=50, color='blue', edgecolors="black")

        # Annotate sample index
        for index, label in enumerate(range(1, len(my_points[:, 0])+1)):
            plt.annotate(label, (my_points[:, 0][index]+x_off, my_points[:, 1][index]+y_off), size=8, style='italic')

        # Make figure to visualize convex hull polygon and highlight polygon formed
        for simplex in hull.simplices:
            plt.plot(my_points[simplex, 0], my_points[simplex, 1], 'r--') # k-
            plt.fill(my_points[hull.vertices, 0], my_points[hull.vertices, 1], c='yellow', alpha=0.01)

        # Aesthetics
        plt.title(title)
        plt.xlabel(Ax)
        plt.ylabel(Ay)
        plt.savefig(title + '.tiff', dpi=300, bbox_inches='tight')
        plt.show()
        return my_points, hull, vertices


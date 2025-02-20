import argparse
import warnings

import numpy as np  # ndarray for gridded data
import pandas as pd  # DataFrames for tabular data

# import utility function
import RigidTransformation_UQI_OOSP as RT  # imports script consisting of functions to run workflow

warnings.filterwarnings('ignore')

Ax1 = 'MDS 1'
Ay1 = 'MDS 2'
x_off1 = 0.025
y_off1 = 0.03



def get_args_parser():
    parser = argparse.ArgumentParser('RigidTransform', add_help=False)

    # model params
    parser.add_argument('--dataset', type=str,
                        default=r'https://raw.githubusercontent.com/GeostatsGuy/GeoDataSets/master/unconv_MV_v4.csv',
                        help="""Location of the data set of interest."""
                        )
    parser.add_argument('-sample_size_start', '--N_start', default=1, type=int, help="""Sample size""")
    parser.add_argument('-sample_size_end', '--N_end', default=100, type=int, help="""Sample size""")
    parser.add_argument('-sample_size_step', '--N_step', default=1, type=int, help="""Sample size""")
    parser.add_argument('--features', default=('PHI (%)', 'AI (kgm2/s)', 'TOC (%)'), type=str, nargs="+",
                        help="""Predictor features """)
    parser.add_argument('--num_response', default='Np (MCFPD)', type=str,
                        help="""Quantitative response feature name """)
    parser.add_argument('--cat_response', default='Np-label', type=str,
                        help="""Categorical response feature name""")
    parser.add_argument('--custom_bins', default=None, type=float,
                        help="""list of 5, starting with 0 used to partition numerical response to categorical""")
    parser.add_argument('--bc_idx', default=0, type=int, help="""Base case index """)
    parser.add_argument('--num_realizations', default=100, type=int, help="""Number of realizations to run code.""")
    parser.add_argument('--base_seed', default=42, type=int, help="""Seed for base case""")
    parser.add_argument('--start_seed', default=1, type=int, help="""Seed starter for realizations""")
    parser.add_argument('--stop_seed', default=10000, type=int, help="""Max seed stopper for realization""")
    parser.add_argument('--idx', default='Well', type=str, help="""Sample name in knowledge domain """)
    parser.add_argument('-dm', '--dissimilarity_metric', default='euclidean', type=str,
                        help="""Dissimilarity metric type from pydist2 package or 'custom' """)
    parser.add_argument('--dim_projection', default='2D', type=str, help="""Dimensionality of LDS""")
    parser.add_argument('--num_OOSP', default=1, type=int, help="""Number of OOSP to add""")
    parser.add_argument('--make_figure', default=False, type=bool, help="""Toggle for convex hull figure""")
    parser.add_argument('--normalize_projections', default=False, type=bool, help="""Ensures scale is homogeneous""")
    parser.add_argument('--custom_dij', default=None,
                       type=float, help="""Custom computed 1D array dissimilarity metric""")
    return parser


# noinspection PyUnusedLocal
def autoresampling(dataframe, N, args):
    ######################
    # data curation
    #####################
    # Add category for response variable i.e., production levels for complete dataset
    df_temp = RT.make_levels(data=dataframe, cat_response=args.cat_response, num_response=args.num_response,
                             custom_bins=args.custom_bins)
    df_subset1 = df_temp.iloc[:N, 1:-1]  # dataframe for n case
    df_subset2, random_seed_oosp = RT.make_sample_within_ci(df_subset1.copy(),
                                                            num_OOSP=args.num_OOSP)  # dataframe for n+1 case
    df_subset2.insert(
        0, args.idx, np.arange(1, len(df_subset2) + 1)
    )  # Insert well column index back into data frame for N+1 case

    df_subset1.insert(
        0, args.idx, np.arange(1, len(df_subset1) + 1)
    )  # Insert well column index back into data frame for N case

    df_subset2 = RT.make_levels(data=df_subset2, cat_response=args.cat_response, num_response=args.num_response,
                                custom_bins=args.custom_bins)

    ######################
    # N case
    #####################
    obj1 = RT.RigidTransformation(df=df_subset1, features=args.features, num_OOSP=args.num_OOSP, idx=args.idx,
                                  num_realizations=args.num_realizations, base_seed=args.base_seed,
                                  start_seed=args.start_seed, stop_seed=args.stop_seed,
                                  dissimilarity_metric=args.dissimilarity_metric, dim_projection=args.dim_projection
                                  , custom_dij=args.custom_dij)

    # Run rigid MDS
    random_seeds, all_real, calc_real, all_rmse, norm_stress = obj1.run_rigid_MDS(
        normalize_projections=args.normalize_projections)
    E1 = obj1.expectation(r_idx=args.bc_idx, Ax=Ax1, Ay=Ay1, verbose=False)

    my_points, hull, vertices = obj1.convex_hull(
        array=all_real, num_OOSP=None, title='N sample case', x_off=0.025, y_off=0.03, Ax=Ax1, Ay=Ay1,
        make_figure=args.make_figure,
        expectation_compute=False, save=False)

    ######################
    # N + 1 case
    #####################
    obj2 = RT.RigidTransf_NPlus(df=df_subset2, features=args.features, num_OOSP=args.num_OOSP, idx=args.idx,
                                num_realizations=args.num_realizations,
                                base_seed=args.base_seed, start_seed=args.start_seed, stop_seed=args.stop_seed,
                                dissimilarity_metric=args.dissimilarity_metric, dim_projection=args.dim_projection
                                , custom_dij=args.custom_dij)

    # Run rigid MDS
    random_seeds2, all_real2, calc_real2, all_rmse2, norm_stress2 = obj2.run_rigid_MDS(
        normalize_projections=args.normalize_projections)  # NEED
    # Find convex hull polygon
    my_points2, hull2, vertices2 = obj2.convex_hull(array=all_real2, num_OOSP=args.num_OOSP, title='N+1 sample case',
                                                    x_off=0.025, y_off=0.03,
                                                    Ax=Ax1, Ay=Ay1, expectation_compute=False, n_case=False,
                                                    save=False, make_figure=args.make_figure)  # 0.05,0.015

    _, _, _, _, rmse_err_anchors, _, _, rmse_err_all = obj2.stabilize_anchors(
        array1=my_points, array2=my_points2, hull_1=hull, hull_2=hull2,
        normalize_projections=args.normalize_projections)  # NEED

    ######################
    # Section 3
    #####################
    E2 = obj2.expectation(r_idx=args.bc_idx, Ax=Ax1, Ay=Ay1, verbose=False)
    my_points_expected, hull_expected, vertices_expected = obj2.convex_hull(
        array=E2, num_OOSP=args.num_OOSP, title='Expectation of N+1 sample case', x_off=x_off1, y_off=y_off1, Ax=Ax1,
        Ay=Ay1,
        expectation_compute=True, n_case=False, save=False, make_figure=args.make_figure
    )

    _, _, _, _, rmse_err_anchors_exp, _, _, rmse_err_all_exp = obj2.stabilize_anchors(
        array1=my_points, array2=my_points_expected, hull_1=hull, hull_2=hull_expected)

    return np.array(list(random_seed_oosp)), np.array(list(random_seeds)), np.array(norm_stress),\
        np.array(norm_stress2), np.array(all_rmse), np.array(all_rmse2), np.array(rmse_err_anchors), \
        np.array(rmse_err_anchors_exp)



if __name__ == '__main__':
    parser_ = argparse.ArgumentParser('RigidTransform', parents=[get_args_parser()])
    args_ = parser_.parse_args()
    df = pd.read_csv(args_.dataset)
    N_VALUES = range(args_.N_start, args_.N_end, args_.N_step)

    # instantiate arrays of zeros
    RandomSeed_oosp = np.zeros((len(N_VALUES), args_.num_OOSP))
    RandomSeed_mds = np.zeros((len(N_VALUES), args_.num_realizations))
    NormStress1 = np.zeros((len(N_VALUES), args_.num_realizations))
    NormStress2 = np.zeros((len(N_VALUES), args_.num_realizations))
    AllRmse1 = np.zeros((len(N_VALUES), args_.num_realizations - 1))
    AllRmse2 = np.zeros((len(N_VALUES), args_.num_realizations - 1))
    RmseAnchors = np.zeros(len(N_VALUES))
    RmseAnchorsExp = np.zeros(len(N_VALUES))

    for index_, n in enumerate(N_VALUES):
        print(f"Processing N = {n}...")
        try:
            oosp_seed, mds_seed, ns1, ns2, rmse1, rmse2, rmse_err1, rmse_err2 = autoresampling(dataframe=df, N=n,
                                                                                               args=args_)
            RandomSeed_oosp[index_] = oosp_seed
            RandomSeed_mds[index_] = mds_seed
            NormStress1[index_] = ns1
            NormStress2[index_] = ns2
            AllRmse1[index_] = rmse1
            AllRmse2[index_] = rmse2
            RmseAnchors[index_] = rmse_err1
            RmseAnchorsExp[index_] = rmse_err2
        except Exception as e:
            print(f"Error occurred at N = {n}: {e}")
            break  # stop the loop if an error occurs

    # Save the lists of arrays to NPY files for 100 realizations at each N-value specified
    np.save('RandomSeed_OOSP.npy', RandomSeed_oosp)  # Random Seed array for OOSP's generated
    np.save('RandomSeed_MDS.npy', RandomSeed_mds)  # Random Seed array for MDS
    np.save('NormStress1_AllReal.npy', NormStress1)  # Normalized kruskal stress1 for N-sample case
    np.save('NormStress2_AllReal.npy', NormStress2)  # Normalized kruskal stress1 for N+1 sample case
    np.save('AllRmse1_AllReal.npy', AllRmse1)  # RMSE for N-sample case
    np.save('AllRmse2_AllReal.npy', AllRmse2)  # RMSE for N+1 sample case
    np.save('RmseAnchors_AllReal.npy', RmseAnchors)  # RMSE for N-sample case anchors
    np.save('RmseAnchorsExp_AllReal.npy', RmseAnchorsExp)  # RMSE for expectation N+1 case anchors

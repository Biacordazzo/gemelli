import unittest
import pandas as pd
import numpy as np
import copy
# from skbio import OrdinationResults
# from pandas import read_csv
# from biom import load_table
# from gemelli.testing import assert_ordinationresults_equal
from gemelli.joint_ctf import (format_time, update_tabular, update_lambda,
                               update_residuals)
# from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal


class TestJointCTF(unittest.TestCase):

    def setUp(self):
        pass

    # def test_concat(self):
    #     """
    #     test concat function inside concat_tensors
    #     """

    # table = process_results[0]
    # metadata = process_results[1]

    # # build the sparse tensor format
    # tensor = build_sparse()
    # tensor.construct(table,
    #                  metadata,
    #                  individual_id_column="SampleID",
    #                  state_column="Timepoint",
    #                  transformation=lambda x: x,
    #                  pseudo_count=0,
    #                  branch_lengths=None,
    #                  replicate_handling="sum",
    #                  svd_centralized=True,
    #                  n_components_centralize=1)

    # tensors[mod_id] = tensor
    # # save all tensors to a class
    # n_tensors = concat_tensors().concat(tensors)

    def test_format_time(self):
        """
        Test Joint-CTF's format_time function
        """

        individual_id_tables = {
            "ind_1": pd.DataFrame(data={
                "sample_1": [1, 0, 2],
                "sample_2": [0, 1, 3],
                "sample_3": [1, 0, 2],
                "sample_4": [1, 0, 2],
                "sample_5": [1, 0, 2]},
                index=["feature_1", "feature_2", "feature_3"]),
            "ind_2": pd.DataFrame(data={
                "sample_1": [4, 1, 0],
                "sample_2": [2, 3, 1],
                "sample_3": [2, 3, 1]},
                index=["feature_1", "feature_2", "feature_3"])}

        individual_id_state_orders = {"ind_1": [1, 2, 3, 4, 5],
                                      "ind_2": [1, 2, 3]}

        # normalized_num = [0, 1, 2, 3, 4]
        # normalized_den = 4

        tables_update = copy.deepcopy(individual_id_tables)

        norm_interval = (0, 1)
        ind_vec = [0, 0, 0, 0, 0, 1, 1, 1]
        Lt = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5]
        ti = [np.array([0, 24, 49, 74, 99]), np.array([0, 24, 49])]

        func_output = format_time(individual_id_tables,
                                  individual_id_state_orders,
                                  n_individuals=2, resolution=100,
                                  input_time_range=(1, 5),
                                  interval=(1, 5))

        self.assertEqual(func_output[0], norm_interval)
        assert_frame_equal(func_output[1]['ind_1'], tables_update['ind_1'])
        assert_frame_equal(func_output[1]['ind_2'], tables_update['ind_2'])
        self.assertEqual(True, np.array_equal(func_output[2][0], ti[0]))
        self.assertEqual(True, np.array_equal(func_output[3], ind_vec))
        self.assertEqual(True, np.array_equal(func_output[4], Lt))

        # now, test with interval
        func_output_2 = format_time(individual_id_tables,
                                    individual_id_state_orders,
                                    n_individuals=2, resolution=100,
                                    input_time_range=(1, 3),
                                    interval=(1, 3))

        # normalized_num = [0, 1, 2, 3, 4]
        # normalized_den = 2

        norm_interval2 = (0, 1)
        ind_vec2 = [0, 0, 0, 0, 0, 1, 1, 1]
        Lt2 = [0.0, 0.5, 1, 1.5, 2, 0.0, 0.5, 1]
        ti2 = [np.array([0, 49, 99]), np.array([0, 49, 99])]

        tables_update2 = {
            "ind_1": pd.DataFrame(data={
                "sample_1": [1, 0, 2],
                "sample_2": [0, 1, 3],
                "sample_3": [1, 0, 2]},
                index=["feature_1", "feature_2", "feature_3"]),
            "ind_2": pd.DataFrame(data={
                "sample_1": [4, 1, 0],
                "sample_2": [2, 3, 1],
                "sample_3": [2, 3, 1]},
                index=["feature_1", "feature_2", "feature_3"])}

        self.assertEqual(func_output_2[0], norm_interval2)
        assert_frame_equal(func_output_2[1]['ind_1'], tables_update2['ind_1'])
        assert_frame_equal(func_output_2[1]['ind_2'], tables_update2['ind_2'])
        self.assertEqual(True, np.array_equal(func_output_2[2][0], ti2[0]))
        self.assertEqual(True, np.array_equal(func_output_2[3], ind_vec2))
        self.assertEqual(True, np.array_equal(func_output_2[4], Lt2))

    def test_update_tabular(self):
        """
        Test Joint-CTF's update_tabular function
        """

        mod1 = {
            "ind_1": pd.DataFrame(data={
                "sample_1": [1, 0, 1],
                "sample_2": [0, 1, 0],
                "sample_3": [2, 3, 2]},
                index=["feature_1", "feature_2", "feature_3"]),
            "ind_2": pd.DataFrame(data={
                "sample_1": [4, 3, 2],
                "sample_2": [1, 3, 3],
                "sample_3": [0, 1, 1]},
                index=["feature_1", "feature_2", "feature_3"])}

        phi_hat = np.array([1, 0, 1])
        beta_hat = np.array([0, 1, 0])
        lambda_ = 10
        ti = [np.array([0, 1, 2]), np.array([0, 1, 2])]

        a_num = {"ind_1": 30, "ind_2": 40}
        a_denom = {"ind_1": 200, "ind_2": 200}
        b_num = np.array([[3, 4],
                          [3, 4],
                          [3, 3]])
        common_denom = {"ind_1": 2, "ind_2": 2}

        joint_ctf_res = update_tabular(mod1, n_individuals=2, n_features=3,
                                       b_mod=beta_hat, phi_mod=phi_hat,
                                       lambda_mod=lambda_, ti=ti)

        self.assertEqual(joint_ctf_res[0]["ind_1"], a_num["ind_1"])
        self.assertEqual(joint_ctf_res[0]["ind_2"], a_num["ind_2"])
        self.assertEqual(joint_ctf_res[1]["ind_1"], a_denom["ind_1"])
        self.assertEqual(joint_ctf_res[1]["ind_2"], a_denom["ind_2"])
        self.assertEqual(True, np.array_equal(joint_ctf_res[2], b_num))
        self.assertEqual(joint_ctf_res[3]["ind_1"], common_denom["ind_1"])
        self.assertEqual(joint_ctf_res[3]["ind_2"], common_denom["ind_2"])

    def test_update_lambda(self):
        """
        Test Joint-CTF's update_lambda function
        """

        mod1 = {
            "ind_1": pd.DataFrame(data={
                "sample_1": [1, 0, 1],
                "sample_2": [0, 1, 0],
                "sample_3": [2, 3, 2]},
                index=["feature_1", "feature_2", "feature_3"]),
            "ind_2": pd.DataFrame(data={
                "sample_1": [4, 2, 2],
                "sample_2": [1, 3, 3],
                "sample_3": [0, 1, 1]},
                index=["feature_1", "feature_2", "feature_3"])}
        phi_hat = np.array([1, 0, 1])
        beta_hat = np.array([0, 1, 0])
        alpha_hat = np.array([30, 60])
        ti = [np.array([0, 1, 2]), np.array([0, 1, 2])]

        lambda_new = update_lambda(mod1, ti=ti, a_hat=alpha_hat,
                                   phi_hat=phi_hat, b_hat=beta_hat)

        self.assertEqual(lambda_new, 0.03)

    def test_update_residuals(self):
        """
        Test Joint-CTF's update_residuals function
        """

        feat_index = ["feature_1", "feature_2"]
        tables = {
            "mod1": {
                "ind_1": pd.DataFrame(data={"sample_1": [4, 10],
                                            "sample_2": [5, 8],
                                            "sample_3": [4, 5]},
                                      index=feat_index)
            },
            "mod2": {
                "ind_1": pd.DataFrame(data={"sample_1": [30, 20],
                                            "sample_2": [25, 15],
                                            "sample_3": [15, 10]},
                                      index=feat_index)
            }
        }

        beta_hats = {"mod1": np.array([1, 2]),
                     "mod2": np.array([3, 2])}
        phi_hats = {"mod1": np.array([1, 2, 1]),
                    "mod2": np.array([2, 1, 1])}
        lambdas = {"mod1": 2, "mod2": 4}
        alpha_hat = np.array([1])  # only one individual in each mod
        ti = {"mod1": [np.array([0, 1, 2]),
                       np.array(["ind_1", "ind_1", "ind_1"])],
              "mod2": [np.array([0, 1, 2]),
                       np.array(["ind_1", "ind_1", "ind_1"])]}

        updated_tables, _ = update_residuals(tables,
                                             a_hat=alpha_hat,
                                             b_hats=beta_hats,
                                             phi_hats=phi_hats,
                                             times=ti,
                                             lambdas=lambdas)

        exp_updates = {
            "mod1": {
                "ind_1": pd.DataFrame(data={"sample_1": [2, 6],
                                            "sample_2": [1, 0],
                                            "sample_3": [2, 1]},
                                      index=feat_index)
            },
            "mod2": {
                "ind_1": pd.DataFrame(data={"sample_1": [6, 4],
                                            "sample_2": [13, 7],
                                            "sample_3": [3, 2]},
                                      index=feat_index)
            }
        }

        assert_frame_equal(updated_tables['mod1']['ind_1'],
                           exp_updates['mod1']['ind_1'])
        assert_frame_equal(updated_tables['mod2']['ind_1'],
                           exp_updates['mod2']['ind_1'])

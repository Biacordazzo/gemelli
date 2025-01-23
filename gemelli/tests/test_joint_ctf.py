import unittest
# import os
# import inspect
import pandas as pd
import numpy as np
import copy
# from skbio import OrdinationResults
# from pandas import read_csv
# from biom import load_table
# from skbio.util import get_data_path
# from gemelli.testing import assert_ordinationresults_equal
# from gemelli.joint_ctf import (update_residuals, get_prop_var, lambda_sort,
#                                reformat_loadings, summation_check,
#                                feature_covariance, update_lambda,
#                                update_a_mod, initialize_tabular,
#                                decomposition_iter, format_time,
#                                formatting_iter, joint_ctf_helper, joint_ctf)
from gemelli.joint_ctf import format_time
# from gemelli.joint_ctf.concat_tensors import concat
# from gemelli.rpca import rpca_table_processing
# from gemelli.preprocessing import build_sparse
# from numpy.testing import assert_allclose


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
        # normalized_t = [0.0, 0.25, 0.5, 0.75, 1.0]

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
        self.assertEqual(func_output[1], tables_update)
        self.assertEqual(func_output[1]['ind_1'].equals(
            tables_update['ind_1']))
        self.assertEqual(func_output[1]['ind_2'].equals(
            tables_update['ind_2']))
        self.assertEqual(func_output[2][0], ti[0])
        self.assertEqual(np.array_equal(func_output[3], ind_vec))
        self.assertEqual(np.array_equal(func_output[4], Lt))

        # now, test with interval
        func_output_2 = format_time(individual_id_tables,
                                    individual_id_state_orders,
                                    n_individuals=2, resolution=100,
                                    input_time_range=(1, 3),
                                    interval=(1, 3))

        # normalized_num = [0, 1, 2, 3, 4]
        # normalized_den = 2
        # normalized_t = [0.0, 0.5, 1, 1.5, 2]

        norm_interval_2 = (0, 1)
        ind_vec_2 = [0, 0, 0, 0, 0, 1, 1, 1]
        Lt_2 = [0.0, 0.5, 1, 1.5, 2, 0.0, 0.5, 1]
        ti_2 = [np.array([0, 49.5, 99]), np.array([0, 49.5, 99])]

        tables_update_2 = {
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

        self.assertEqual(func_output_2[0], norm_interval_2)
        self.assertEqual(func_output_2[1], tables_update_2)
        self.assertEqual(func_output_2[1]['ind_1'].equals(
            tables_update_2['ind_1']))
        self.assertEqual(func_output_2[1]['ind_2'].equals(
            tables_update_2['ind_2']))
        self.assertEqual(func_output_2[2][0], ti_2[0])
        self.assertEqual(np.array_equal(func_output_2[3], ind_vec_2))
        self.assertEqual(np.array_equal(func_output_2[4], Lt_2))

    def test_formatting_iter(self):
        pass

    def test_decomposition_iter(self):
        pass

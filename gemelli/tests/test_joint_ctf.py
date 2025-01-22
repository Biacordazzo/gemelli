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
        test format_time function
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
                "sample_4": [2, 3, 1]},
                index=["feature_1", "feature_2", "feature_3"])}
        
        individual_id_state_orders = {"ind_1": [1, 2, 3, 4, 5],
                                      "ind_2": [1, 2, 3]}
        
        # normalized_num = [0, 1, 2, 3, 4]
        # normalized_den = 4
        # normalized_t = [0.0, 0.25, 0.5, 0.75, 1.0]
        # resolution = 100
        # input_time_range = (1, 5)
        # interval = (1, 5)

        tables_update = copy.deepcopy(individual_id_tables)

        norm_interval = (0, 1)
        ind_vec = [0,0,0,0,0,1,1,1]
        Lt = [[0.0, 24.75, 49.5, 74.25, 99.0, 0.0, 24.75, 49.5]]
        ti = [np.array([0, 24, 49, 74, 99]), np.array([0, 24, 49])]

        func_output = format_time(individual_id_tables, 
                                  individual_id_state_orders, 
                                  n_individuals=2, resolution=100,
                                  input_time_range=(0,5),
                                  interval=None)

        self.assertEqual(func_output, 
                        (norm_interval, tables_update, ti, ind_vec, Lt))

    def test_formatting_iter(self):
        pass

    def test_decomposition_iter(self):
        pass

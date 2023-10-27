# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class Path:
    @staticmethod
    def db_root_dir(db_name):
        if db_name == "CUB":
            return "/home/marcin.przewiezlikowki/datasets/CUB_200_2011/"
        if db_name == "CUB_framed":
            # return "/shared/sets/datasets/birds/"
            return "/home/marcin.przewiezlikowki/uj/proto_plugen/checkpoints"
        else:
            raise NotImplementedError

    @staticmethod
    def output_root_dir():
        # return "/home/przewiez/uj/visual-counterfactuals/results"
        # return "/home/mprzewie/coding/uj/visual-counterfactuals/results/"
        return "/home/marcin.przewiezlikowki/uj/visual-counterfactuals/results"
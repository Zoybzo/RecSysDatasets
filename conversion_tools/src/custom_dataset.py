import bz2
import csv
import json
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from base_dataset import BaseDataset
from cosmetics import CosmeticsDataset


class YELPDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(YELPDataset, self).__init__(input_path, output_path)
        self.dataset_name = "yelp"

        # input file
        self.inter_file = os.path.join(
            self.input_path, "yelp_academic_dataset_review.json"
        )
        self.item_file = os.path.join(
            self.input_path, "yelp_academic_dataset_business.json"
        )
        self.user_file = os.path.join(
            self.input_path, "yelp_academic_dataset_user.json"
        )

        # output_file
        self.output_inter_file, self.output_item_file, self.output_user_file = (
            self.get_output_files()
        )
        output_user2index_file = os.path.join(self.output_path, self.dataset_name + 'user2index')
        output_item2index_file = os.path.join(self.output_path, self.dataset_name + 'item2index')
        self.output_user2index_file, self.output_item2index_file = output_user2index_file, output_item2index_file

        # selected feature fields
        self.inter_fields = {
            0: "review_id:token",
            1: "user_id:token",
            2: "business_id:token",
            3: "stars:float",
            4: "useful:float",
            5: "funny:float",
            6: "cool:float",
            8: "date:float",
        }

        self.item_fields = {
            0: "business_id:token",
            1: "item_name:token_seq",
            2: "address:token_seq",
            3: "city:token_seq",
            4: "state:token",
            5: "postal_code:token",
            6: "latitude:float",
            7: "longitude:float",
            8: "item_stars:float",
            9: "item_review_count:float",
            10: "is_open:float",
            12: "categories:token_seq",
        }

        self.user_fields = {
            0: "user_id:token",
            1: "name:token",
            2: "review_count:float",
            3: "yelping_since:float",
            4: "useful:float",
            5: "funny:float",
            6: "cool:float",
            7: "elite:token",
            9: "fans:float",
            10: "average_stars:float",
            11: "compliment_hot:float",
            12: "compliment_more:float",
            13: "compliment_profile:float",
            14: "compliment_cute:float",
            15: "compliment_list:float",
            16: "compliment_note:float",
            17: "compliment_plain:float",
            18: "compliment_cool:float",
            19: "compliment_funny:float",
            20: "compliment_writer:float",
            21: "compliment_photos:float",
        }
        self.user_head_fields = {
            0: "user_id:token",
            1: "user_name:token",
            2: "user_review_count:float",
            3: "yelping_since:float",
            4: "user_useful:float",
            5: "user_funny:float",
            6: "user_cool:float",
            7: "elite:token",
            9: "fans:float",
            10: "average_stars:float",
            11: "compliment_hot:float",
            12: "compliment_more:float",
            13: "compliment_profile:float",
            14: "compliment_cute:float",
            15: "compliment_list:float",
            16: "compliment_note:float",
            17: "compliment_plain:float",
            18: "compliment_cool:float",
            19: "compliment_funny:float",
            20: "compliment_writer:float",
            21: "compliment_photos:float",
        }

    def load_item_data(self):
        return pd.read_json(self.item_file, lines=True)

    def convert_inter(self):
        fin = open(self.inter_file, "r")
        fout = open(self.output_inter_file, "w")

        lines_count = 0
        for _ in fin:
            lines_count += 1
        fin.seek(0, 0)

        fout.write(
            "\t".join(
                [self.inter_fields[column] for column in self.inter_fields.keys()]
            )
            + "\n"
        )

        user_map = {}
        item_map = {}
        for i in tqdm(range(lines_count)):
            line = fin.readline()
            line_dict = json.loads(line)
            line_dict["date"] = int(
                time.mktime(time.strptime(line_dict["date"], "%Y-%m-%d %H:%M:%S"))
            )
            # Update the stars
            def get_behavior(stars):
                if stars <= 2:
                    return 0 # dislike
                elif stars >= 4:
                    return 2 # like
                else:
                    return 1 # neutral
            line_dict["stars"] = get_behavior(line_dict["stars"])
            # Update the user_id
            if line_dict["user_id"] not in user_map:
                user_map["user_id"] = len(user_map) + 1
            line_dict['user_id'] = user_map[line_dict['user_id']]
            if line_dict["business_id"] not in item_map:
                item_map['business_id'] = len(item_map) + 1
            line_dict['business_id'] = item_map[line_dict['business_id']]
            fout.write(
                "\t".join(
                    [
                        str(
                            line_dict[
                                self.inter_fields[key][
                                    0 : self.inter_fields[key].find(":")
                                ]
                            ]
                        )
                        for key in self.inter_fields.keys()
                    ]
                )
                + "\n"
            )

        fin.close()
        fout.close()
        with open(self.output_user2index_file, 'w') as f:
            for k, v in user_map.items():
                f.write(f"{k}\t{v}\n")
        with open(self.output_item2index_file, 'w') as f:
            for k, v in item_map.items():
                f.write(f"{k}\t{v}\n")

    def convert_user(self):
        fin = open(self.user_file, "r")
        fout = open(self.output_user_file, "w")

        lines_count = 0
        for _ in fin:
            lines_count += 1
        fin.seek(0, 0)

        fout.write(
            "\t".join(
                [
                    self.user_head_fields[column]
                    for column in self.user_head_fields.keys()
                ]
            )
            + "\n"
        )

        for i in tqdm(range(lines_count)):
            line = fin.readline()
            line_dict = json.loads(line)
            line_dict["yelping_since"] = int(
                time.mktime(
                    time.strptime(line_dict["yelping_since"], "%Y-%m-%d %H:%M:%S")
                )
            )
            fout.write(
                "\t".join(
                    [
                        str(
                            line_dict[
                                self.user_fields[key][
                                    0 : self.user_fields[key].find(":")
                                ]
                            ]
                        )
                        for key in self.user_fields.keys()
                    ]
                )
                + "\n"
            )

        fin.close()
        fout.close()

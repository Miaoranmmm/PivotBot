import datasets
import jsonlines
import os

# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Corpus for MultiwozChat"""


import csv

import datasets

# from datasets import dataclass

_DESCRIPTION = """\
OBMultiWOZ
"""

_CITATION = """\
OBMultiWOZ
"""

_DOWNLOAD_URL = ""
_WEBPAGE = ""


class NewConfig(datasets.BuilderConfig):
    """BuilderConfig for Task."""
    def __init__(self, name, task, start_idx, fewshot, **kwargs) -> None:
        super(NewConfig).__init__()
        self.task = task
        self.name = name
        self.fewshot = fewshot
        self.start = start_idx



class multiwoz_chat_no_kw(datasets.GeneratorBasedBuilder):
    """mmultiwoz"""
    names = ['init_chitchat', 'insert_chitchat', 'multi_insert']
    tasks = ['taskbot', 'chitchat', 'mix']
    fewshots = [100, 200, 500]
    start_index = [0, 100, 200, 300]
    BUILDER_CONFIGS = []
    for name in names:
        for task in tasks:
            for fewshot in fewshots:
                if fewshot == 500:
                    BUILDER_CONFIGS.append(
                            NewConfig(
                                #name=name + '-' + task + '-' + str(fewshot),
                                name=name + '-' + task + '-' + '0' + '-' + str(fewshot) ,
                                task=task,
                                start_idx=0,
                                fewshot=fewshot)
                            )
                else:
                    for start_idx in start_index:
                        BUILDER_CONFIGS.append(
                                NewConfig(
                                    #name=name + '-' + task + '-' + str(fewshot),
                                    name=name + '-' + task + '-' + str(start_idx) + '-' + str(fewshot) ,
                                    task=task,
                                    start_idx=start_idx,
                                    fewshot=fewshot)
                                )
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Context": datasets.Value("string"),
                    "Response": datasets.Value("string"),
                    "Knowledge": datasets.Value("string"),
                    "Selected_knowledge": datasets.Value("string"),
                    "Id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Query": datasets.Value("string"),
                    
                }
            ),
            homepage=_WEBPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        base_dir = 'data'
        data_dir = os.path.join('data', self.config.name.split('-')[0])

        train_path = [os.path.join(data_dir, 'train_new_no_kw.jsonl')]
        train_flist = os.path.join(base_dir, 'trainFile_new.txt')
        validation_path = [os.path.join(data_dir, 'valid_new_no_kw.jsonl')]
        validation_flist = os.path.join(base_dir, 'validFile_new.txt')
        test_path = [os.path.join(data_dir, 'test_new_no_kw.jsonl')]
        test_flist = os.path.join(base_dir, 'testFile_new.txt')
        return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path, "filelist": train_flist, "stage": "train"}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": validation_path, "filelist": validation_flist, "stage": "valid"}),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path, "filelist": test_flist, "stage": "test"}),
        ]


    def _generate_examples(self, filepath, filelist, stage):
        key = 0
        files = set()
        f_names = open(filelist, 'r').read().splitlines()
        if stage == 'train':
            selected_files = f_names[self.config.start : self.config.start + self.config.fewshot]
        else:
            selected_files = f_names

        for filename in filepath:
            with open(filename, "r", encoding="utf-8") as reader:  
                for item in jsonlines.Reader(reader):
#                    if len(files) > self.config.fewshot:
#                        continue
#                    elif len(files) <= 100:
#                        files.add(item['Id'])
#                        continue
                    if item['Id'] not in selected_files:
                        continue

                    if self.config.task == 'mix': 
                        files.add(item['Id'])
                        yield key, item
                        key += 1
                    elif item['Task'] == self.config.task:
                        files.add(item['Id'])
                        yield key, item
                        key += 1

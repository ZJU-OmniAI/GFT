# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import Union

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class GFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config):
        # 初始化基本信息,prompt、answer、question的key值索引，方便寻找
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        response_dict_keys_model = config.get("response_dict_keys_model", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]# 如果为False，会抛出异常
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer # 类型注解+赋值，注意类型注解不影响实际执行
        
        # 获取key
        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []
        self.response_dict_keys_model = response_dict_keys_model if response_dict_keys_model else []

        self.max_length = max_length
        

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls.iloc[0]
            return ls

        # 1. 读取数据
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        
        # 2.获取prompts(问题)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # 提取纯问题 noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        
        # 3.获取response -> ground_Truth
        self.responses_groundTruth = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses_groundTruth = self.responses_groundTruth.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses_groundTruth}")
                raise
        if isinstance(self.responses_groundTruth, pd.DataFrame):
            self.responses_groundTruth = self.responses_groundTruth.squeeze()
        self.responses_groundTruth = self.responses_groundTruth.tolist()
        
        # 4.获取response -> model_output
        self.responses_model = self.dataframe[self.response_key]
        for key in self.response_dict_keys_model:
            try:
                self.responses_model = self.responses_model.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses_model}")
                raise
        if isinstance(self.responses_model, pd.DataFrame):
            self.responses_model = self.responses_model.squeeze()
        self.responses_model = self.responses_model.tolist()
        
        # 5.根据config 选取最终的回答
        ### 暂时就先选取一条ground_truth + 三条model_output 作为数据集
        self.responses = []
        for i in range(len(self.responses_groundTruth)):
            new_response = [
                x for x in self.responses_model[i][:3]
            ]
            new_response.append(self.responses_groundTruth[i])
            
            self.responses.append(new_response)
        
        del self.responses_model
        del self.responses_groundTruth
        
        
        

    def __len__(self):
        return len(self.prompts)

    # def __getitem__(self, item):
    #     # 获取tokenzier,prompt(问题),response(回答)
    #     tokenizer = self.tokenizer

    #     prompt = self.prompts[item]
    #     response = self.responses[item]
    #     response = response[0]

    #     # apply chat template 构造聊天模板
    #     prompt_chat = [{"role": "user", "content": prompt}]

    #     # string 将聊天模板(含问题)转换为str
    #     prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
    #     response_chat_str = response + tokenizer.eos_token

    #     # tokenize 对问题进行tokenizer获取到ids和attention_mask
    #     prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
    #     prompt_ids = prompt_ids_output["input_ids"][0]
    #     prompt_attention_mask = prompt_ids_output["attention_mask"][0]
    #     # tokenize 对问题进行tokenizer获取到ids和attention_mask
    #     response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
    #     response_ids = response_ids_output["input_ids"][0]
    #     response_attention_mask = response_ids_output["attention_mask"][0]

    #     prompt_length = prompt_ids.shape[0]
    #     response_length = response_ids.shape[0]

    #     # 整理最终的input_ids和attention_mask
    #     input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    #     attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

    #     # padding to max length
    #     sequence_length = input_ids.shape[0]
    #     if sequence_length < self.max_length:
    #         padded_input_ids = (
    #             torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
    #             * self.tokenizer.pad_token_id
    #         )
    #         padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

    #         input_ids = torch.cat((input_ids, padded_input_ids))
    #         attention_mask = torch.cat((attention_mask, padded_attention_mask))
    #     elif sequence_length > self.max_length:
    #         if self.truncation == "left":
    #             # actually, left truncation may not be reasonable
    #             input_ids = input_ids[-self.max_length :]
    #             attention_mask = attention_mask[-self.max_length :]
    #         elif self.truncation == "right":
    #             input_ids = input_ids[: self.max_length]
    #             attention_mask = attention_mask[: self.max_length]
    #         elif self.truncation == "error":
    #             raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
    #         else:
    #             raise NotImplementedError(f"Unknown truncation method {self.truncation}")

    #     position_ids = compute_position_id_with_mask(attention_mask)

    #     loss_mask = attention_mask.clone()
    #     if prompt_length > 1:
    #         # mask out prompt for SFT.
    #         loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
    #     # mask out the last token in response
    #     loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "position_ids": position_ids,
    #         "loss_mask": loss_mask,
    #     }
        
    def __getitem__(self, item):
        # 获取tokenzier,prompt(问题)
        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        
        # 1. responses 现在是一个包含多条回答的列表
        responses_list = self.responses[item]

        # --- 处理问题部分 (对所有回答都相同,只需要执行一次) ---
        
        # apply chat template 构造聊天模板
        prompt_chat = [{"role": "user", "content": prompt}]

        # string 将聊天模板(含问题)转换为str
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        
        # tokenize 对问题进行tokenizer获取到ids和attention_mask
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        prompt_length = prompt_ids.shape[0]

        # --- 初始化列表 ---
        # 用于存储每个 <问题, 答案> 对的处理结果
        all_input_ids = []
        all_attention_masks = []
        all_position_ids = []
        all_loss_masks = []

        # 2. 遍历列表中的每一个回答
        for response in responses_list:
            # --- (循环内部) 处理单个回答 ---
            
            # response = response[0] # 原有代码, 已被外层循环替代
            response_chat_str = response + tokenizer.eos_token

            # tokenize 对(单个)回答进行tokenizer获取到ids和attention_mask
            response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
            response_ids = response_ids_output["input_ids"][0]
            response_attention_mask = response_ids_output["attention_mask"][0]
            response_length = response_ids.shape[0]

            # 整理最终的input_ids和attention_mask
            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

            # padding to max length
            sequence_length = input_ids.shape[0]
            if sequence_length < self.max_length:
                padded_input_ids = (
                    torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                    * self.tokenizer.pad_token_id
                )
                padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    # actually, left truncation may not be reasonable
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                elif self.truncation == "error":
                    raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise NotImplementedError(f"Unknown truncation method {self.truncation}")

            # (循环内部) 计算 position_ids
            position_ids = compute_position_id_with_mask(attention_mask)

            # (循环内部) 计算 loss_mask
            loss_mask = attention_mask.clone()
            if prompt_length > 1:
                # mask out prompt for SFT.
                loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
            # mask out the last token in response
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

            # --- (循环内部) 将处理好的tensor添加到列表中 ---
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_position_ids.append(position_ids)
            all_loss_masks.append(loss_mask)

        # 3. 循环结束后, 将列表中的Tensors堆叠成一个新的Tensor
        #    torch.stack(..., dim=0) 会在第0维增加一个新的维度
        #    最终shape为 (N, max_length), N是回答的数量
        final_input_ids = torch.stack(all_input_ids, dim=0)
        final_attention_mask = torch.stack(all_attention_masks, dim=0)
        final_position_ids = torch.stack(all_position_ids, dim=0)
        final_loss_mask = torch.stack(all_loss_masks, dim=0)

        # 返回堆叠后的Tensors字典
        # 这样 final_input_ids[i] 就对应 final_loss_mask[i]
        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "position_ids": final_position_ids,
            "loss_mask": final_loss_mask,
            "length": prompt_length
        }

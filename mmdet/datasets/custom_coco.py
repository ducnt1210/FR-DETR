import os.path as osp
from typing import List, Union

import torch
import mmcv
import numpy as np
from mmengine.fileio import get_local_path
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from safetensors import safe_open

@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    """COCO dataset with mask and descriptions"""

    METAINFO = {
        'classes': ('car', 'bus', 'motorcycle', 'bicycle', 'person', 'truck'),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }

    def __init__(self, mask_dir: str, tensor_file: str, *args, **kwargs):
        self.mask_dir = mask_dir
        self.tensor_file = tensor_file
        # print(1) # debug 1
        if self.tensor_file != None:
            with safe_open(self.tensor_file, framework="pt", device="cpu") as f:
                self.tensors = {key: f.get_tensor(key) for key in f.keys()}
        else:
            self.tensors = {}
        # print(2) # debug 2
        super().__init__(*args, **kwargs)

        # Load all tensors from the safetensors file

    def load_data_list(self) -> List[dict]:
        return super().load_data_list()

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]: # handle for split weather condition and object class.
        data_info = super().parse_data_info(raw_data_info)

        # Load mask image
        img_filename = osp.basename(data_info['img_path'])
        mask_filename = osp.splitext(img_filename)[0] + '_mask_0.jpg'  # Adjust this pattern if needed
        if self.mask_dir != None:
            mask_path = osp.join(self.mask_dir, mask_filename)
        else:
            mask_path = None
        
        data_info['mask_path'] = mask_path

        # # Load mask image
        # if osp.exists(mask_path):
        #     mask = mmcv.imread(mask_path, flag='grayscale')
        #     mask = np.where(mask > 128, 1, 0)
        #     data_info['object_mask'] = mask
        # else:
        #     data_info['object_mask'] = np.zeros((data_info['img'].shape[1], data_info['img'].shape[2]), dtype=np.float32)

        # Get tensor for this image
        tensor_key = img_filename
        data_info['embedding_1'] = self.tensors.get(f"{tensor_key}_weather", None)
        if data_info['embedding_1'] is None:
            print(f"Warning: {tensor_key}_weather is None")

        data_info['embedding_2'] = self.tensors.get(f"{tensor_key}_object", None)
        if data_info['embedding_2'] is None:
            print(f"Warning: {tensor_key}_object is None")


        # if data_info['embedding'] is not None:
        #     embedding = torch.as_tensor(data_info['embedding'])
            
        #     # Ensure the embedding has 3 dimensions
        #     if embedding.dim() == 2:
        #         embedding = embedding.unsqueeze(0)
            
        #     # Get the current shape
        #     current_length = embedding.shape[1]
            
        #     if current_length < 455:
        #         padding = torch.zeros(1, 455 - current_length, embedding.shape[2])
        #         data_info['embedding'] = torch.cat([embedding, padding], dim=1)
        #     elif current_length > 455:
        #         data_info['embedding'] = embedding[:, :455, :]
        #     else:
        #         data_info['embedding'] = embedding
            
        # else:
        #     data_info['embedding'] = torch.zeros(1, 455, 5120)

        return data_info

    def prepare_data(self, idx: int) -> Union[dict, List[dict]]:
        max_seq_len_1 = 10
        max_seq_len_2 = 15
        feat_dim = 768 # Bert

        data_info = self.get_data_info(idx)
        # print("This is data_info from custom_coco", data_info)
        # raise SystemExit

        # Load mask image
        if data_info['mask_path'] is not None:
            if osp.exists(data_info['mask_path']):
                mask = mmcv.imread(data_info['mask_path'], flag='grayscale')
                # mask = np.where(mask > 128, 1, 0).astype(dtype=np.float32)
                mask = mask.astype(np.float32) / 255
                data_info['object_mask'] = mask
            else:
                print("No path for the image")
                print(data_info['mask_path'])
                data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)
        else:
            print("Mask path is None")
            data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)

        # Ensure embedding_1 and embedding_2 are tensors and pad/truncate to [1, max_seq_len, feat_dim]
        if data_info['embedding_1'] is not None and data_info['embedding_2'] is not None:
            embedding_1 = torch.as_tensor(data_info['embedding_1'])
            embedding_2 = torch.as_tensor(data_info['embedding_2'])
            
            # Ensure both embeddings have 3 dimensions
            if embedding_1.dim() == 2:
                embedding_1 = embedding_1.unsqueeze(0)
            if embedding_2.dim() == 2:
                embedding_2 = embedding_2.unsqueeze(0)
            
            # Get the current shape for each embedding
            current_length_1 = embedding_1.shape[1]
            current_length_2 = embedding_2.shape[1]

            # Pad or truncate embedding_1
            embedding_1, mask_1 = pad_or_truncate(embedding_1, max_seq_len_1, feat_dim)

            # Pad or truncate embedding_2
            embedding_2, mask_2 = pad_or_truncate(embedding_2, max_seq_len_2, feat_dim)

            # Concatenate embedding_1 and embedding_2
            key_embedding = torch.cat([embedding_1, embedding_2], dim=1)

            # Concatenate masks
            combined_mask = torch.cat([mask_1, mask_2], dim=1)
            
        else:
            # Handle the case where one or both embeddings are None
            key_embedding = torch.zeros(1, max_seq_len_1 + max_seq_len_2, feat_dim)
            combined_mask = torch.zeros(1, max_seq_len_1 + max_seq_len_2)

        # Store the final key embedding and mask
        data_info['embedding'] = key_embedding
        data_info['embedding_mask'] = combined_mask

        del data_info['embedding_1']
        del data_info['embedding_2']

        # Add mask and embedding tensors to results
        data_info = self.pipeline(data_info)

        if self.test_mode:
            if not torch.is_tensor(data_info['data_samples'].gt_instances.bboxes):
                data_info['data_samples'].gt_instances.bboxes = data_info['data_samples'].gt_instances.bboxes.tensor
            if not torch.is_tensor(data_info['data_samples'].ignored_instances.bboxes):
                data_info['data_samples'].ignored_instances.bboxes = data_info['data_samples'].ignored_instances.bboxes.tensor
            temp_pad_shape = (data_info['data_samples'].img_shape[0], data_info['data_samples'].img_shape[1])
            data_info['data_samples'].set_metainfo({
                    'batch_input_shape': temp_pad_shape,
                    'pad_shape': temp_pad_shape
                })

        # print("This is data_info in customCocoDataset", data_info)
        # raise SystemExit()

        return data_info

    # def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]: # handle for v1 text with mask and language
    #     data_info = super().parse_data_info(raw_data_info)

    #     # Load mask image
    #     img_filename = osp.basename(data_info['img_path'])
    #     mask_filename = osp.splitext(img_filename)[0] + '_mask_0.jpg'  # Adjust this pattern if needed
    #     mask_path = osp.join(self.mask_dir, mask_filename)
    #     data_info['mask_path'] = mask_path

    #     tensor_key = img_filename
    #     data_info['embedding'] = self.tensors.get(tensor_key, None)

    #     return data_info

    # def prepare_data(self, idx: int) -> Union[dict, List[dict]]:
    #     data_info = self.get_data_info(idx)
    #     # print("This is data_info from custom_coco", data_info)
    #     # raise SystemExit

    #     # Load mask image
    #     if data_info['mask_path'] is not None:
    #         if osp.exists(data_info['mask_path']):
    #             mask = mmcv.imread(data_info['mask_path'], flag='grayscale')
    #             # mask = np.where(mask > 128, 1, 0).astype(dtype=np.float32)
    #             mask = mask.astype(np.float32) / 255
    #             data_info['object_mask'] = mask
    #         else:
    #             # print("No path for the image")
    #             data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)
    #     else:
    #         # print("Mask path is None")
    #         data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)

    #     # Ensure embedding is a tensor and pad/truncate to [1, 455, 5120]
    #     if data_info['embedding'] is not None:
    #         embedding = torch.as_tensor(data_info['embedding'])
            
    #         # Ensure the embedding has 3 dimensions
    #         if embedding.dim() == 2:
    #             embedding = embedding.unsqueeze(0)
            
    #         # Get the current shape
    #         current_length = embedding.shape[1]
            
    #         max_length = 10
    #         embed_dim = 768
    #         if current_length < max_length:
    #             padding = torch.zeros(1, max_length - current_length, embedding.shape[2])
    #             data_info['embedding'] = torch.cat([embedding, padding], dim=1)
    #             data_info['embedding_mask'] = torch.cat([torch.ones(1, current_length), torch.zeros(1, max_length - current_length)], dim=1)
    #         elif current_length > max_length:
    #             data_info['embedding'] = embedding[:, :max_length, :]
    #             data_info['embedding_mask'] = torch.ones(1, max_length)
    #         else:
    #             data_info['embedding'] = embedding
    #             data_info['embedding_mask'] = torch.ones(1, max_length)
            
    #     else:
    #         data_info['embedding'] = torch.zeros(1, max_length, embed_dim)
    #         data_info['embedding_mask'] = torch.zeros(1, max_length)

    #     # Add mask and embedding tensors to results
    #     data_info = self.pipeline(data_info)

    #     if self.test_mode:
    #         if not torch.is_tensor(data_info['data_samples'].gt_instances.bboxes):
    #             data_info['data_samples'].gt_instances.bboxes = data_info['data_samples'].gt_instances.bboxes.tensor
    #         if not torch.is_tensor(data_info['data_samples'].ignored_instances.bboxes):
    #             data_info['data_samples'].ignored_instances.bboxes = data_info['data_samples'].ignored_instances.bboxes.tensor
    #         temp_pad_shape = (data_info['data_samples'].img_shape[0], data_info['data_samples'].img_shape[1])
    #         data_info['data_samples'].set_metainfo({
    #                 'batch_input_shape': temp_pad_shape,
    #                 'pad_shape': temp_pad_shape
    #             })

    #     # print("This is data_info in customCocoDataset", data_info)
    #     # raise SystemExit()

    #     return data_info

    # def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]: # handle for sentence transformer text
    #     data_info = super().parse_data_info(raw_data_info)

    #     # Load mask image
    #     img_filename = osp.basename(data_info['img_path'])
    #     mask_filename = osp.splitext(img_filename)[0] + '_mask_0.jpg'  # Adjust this pattern if needed
    #     mask_path = osp.join(self.mask_dir, mask_filename)
    #     data_info['mask_path'] = mask_path

    #     # Get tensor for this image
    #     tensor_key = img_filename
    #     if tensor_key in self.tensors:
    #         data_info['embedding'] = self.tensors[tensor_key]
    #     else:
    #         print("Can not find tensor key in safetensors")
    #         data_info['embedding'] = None

    #     return data_info

    # def prepare_data(self, idx: int) -> Union[dict, List[dict]]:
    #     data_info = self.get_data_info(idx)

    #     # Load mask image
    #     # Load mask image
    #     if data_info['mask_path'] is not None:
    #         if osp.exists(data_info['mask_path']):
    #             mask = mmcv.imread(data_info['mask_path'], flag='grayscale')
    #             # mask = np.where(mask > 128, 1, 0).astype(dtype=np.float32)
    #             mask = mask.astype(np.float32) / 255
    #             data_info['object_mask'] = mask
    #         else:
    #             print("No path for the image")
    #             data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)
    #     else:
    #         print("Mask path is None")
    #         data_info['object_mask'] = np.zeros((data_info['width'], data_info['height']), dtype=np.float32)

    #     # Ensure embedding is a tensor and always have length = 1 and embedding dim = 768
    #     if data_info['embedding'] is not None:
    #         embedding = torch.as_tensor(data_info['embedding'])
            
    #         # Ensure the embedding has 3 dimensions
    #         if embedding.dim() == 2:
    #             embedding = embedding.unsqueeze(0)
            
    #         # Fix embedding to have shape [1, 1, 768]
    #         data_info['embedding'] = embedding[:, :1, :768]  # Truncate to [1, 1, 768] if necessary
            
    #         # Create a corresponding embedding mask of shape [1, 1]
    #         data_info['embedding_mask'] = torch.ones(1, 1, dtype=torch.float32)
    #     else:
    #         # Handle the case where embedding is None
    #         data_info['embedding'] = torch.zeros(1, 1, 768)
    #         data_info['embedding_mask'] = torch.zeros(1, 1, dtype=torch.float32)

    #     # Add mask and embedding tensors to results
    #     data_info = self.pipeline(data_info)

    #     if self.test_mode:
    #         if not torch.is_tensor(data_info['data_samples'].gt_instances.bboxes):
    #             data_info['data_samples'].gt_instances.bboxes = data_info['data_samples'].gt_instances.bboxes.tensor
    #         if not torch.is_tensor(data_info['data_samples'].ignored_instances.bboxes):
    #             data_info['data_samples'].ignored_instances.bboxes = data_info['data_samples'].ignored_instances.bboxes.tensor
    #         temp_pad_shape = (data_info['data_samples'].img_shape[0], data_info['data_samples'].img_shape[1])
    #         data_info['data_samples'].set_metainfo({
    #                 'batch_input_shape': temp_pad_shape,
    #                 'pad_shape': temp_pad_shape
    #             })

    #     return data_info

def pad_or_truncate(embedding, max_seq_len, feat_dim):
        current_length = embedding.shape[1]
        
        if current_length < max_seq_len:
            padding = torch.zeros(1, max_seq_len - current_length, feat_dim)
            padded_embedding = torch.cat([embedding, padding], dim=1)
            mask = torch.cat([torch.ones(1, current_length), torch.zeros(1, max_seq_len - current_length)], dim=1)
        elif current_length > max_seq_len:
            padded_embedding = embedding[:, :max_seq_len, :]
            mask = torch.ones(1, max_seq_len)
        else:
            padded_embedding = embedding
            mask = torch.ones(1, max_seq_len)
        
        return padded_embedding, mask
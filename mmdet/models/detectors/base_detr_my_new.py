# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

from torch import Tensor
import torch

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.optim import OptimWrapper
from mmdet.structures import DetDataSample
from .base import BaseDetector
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

@MODELS.register_module()
class MyNewDetectionTransformer(BaseDetector, metaclass=ABCMeta):
    r"""Base class for Detection Transformer.

    In Detection Transformer, an encoder is used to process output features of
    neck, then several queries interact with the encoder features using a
    decoder and do the regression and classification with the bounding box
    head.

    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 restormer:ConfigType,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.restormer = MODELS.build(restormer)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

    @abstractmethod
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        pass

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             masks: Tensor,
             embeddings: Tensor,
             embedding_masks: Tensor) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            masks (Tensor): Input segmentation mask for the images of shape (N, 1, H, W)
            embeddings: (Tensor) Description sequence of the images by LLM (N, 1, seq_length, feature dimension) (Bert: feature dim = 768) Ex: [2, 1, 300, 768]
            embed_masks: (Tensor) Mask of the embeddings (N, 1, seq_length)

        Returns:
            dict: A dictionary of loss components
        """
        restored_fea= self.restormer(batch_inputs, embeddings, embedding_masks)
        img_feats = self.extract_feat(restored_fea)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                masks: Tensor,
                embeddings: Tensor,
                embedding_masks: Tensor,
                rescale: bool = True,
                return_enhanced_images: bool = False,
                normalize_input: bool = False) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            masks (Tensor): Input segmentation mask for the images of shape (N, 1, H, W)
            embeddings: (Tensor) Description sequence of the images by LLM (N, 1, seq_length, feature dimension) (Bert: feature dim = 768) Ex: [2, 1, 300, 768]
            embed_masks: (Tensor) Mask of the embeddings (N, 1, seq_length)
            rescale (bool): Whether to rescale the results.
                Defaults to True.
            return_enhanced_images (bool): Return the enhanced image after unnormalized. 
            normalize_input (bool): normalize input with data preprocessor std and mean.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if normalize_input:
            batch_inputs = self.normalize(batch_inputs)

        restored_fea= self.restormer(batch_inputs, embeddings, embedding_masks)
        img_feats = self.extract_feat(restored_fea)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        if return_enhanced_images:
            return self.unnormalize(restored_fea), batch_data_samples 
        
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            masks: Tensor,
            embeddings: Tensor,
            embedding_masks: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        restored_fea= self.restormer(batch_inputs, embeddings, embedding_masks)
        img_feats = self.extract_feat(restored_fea)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @abstractmethod
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              'feat_pos', and other algorithm-specific arguments.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask', and
              other algorithm-specific arguments.
        """
        pass

    @abstractmethod
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, **kwargs) -> Dict:
        """Forward with Transformer encoder.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output and other algorithm-specific
            arguments.
        """
        pass

    @abstractmethod
    def pre_decoder(self, memory: Tensor, **kwargs) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and other algorithm-specific arguments.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """
        pass

    @abstractmethod
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        **kwargs) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output, `references` including
            the initial and intermediate reference_points, and other
            algorithm-specific arguments.
        """
        pass

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
                input from this dict will contain list of images, masks, embeddings and embedding masks.
                Shape after stack (all will be tensor)
                    img: (N, C, H, W) Ex: [2, 3, 360, 640]
                    mask: (N, 1, H, W)
                    embedding: (N, 1, seq_length, feature dimension) (Bert: feature dim = 768) Ex: [2, 1, 300, 768]
                    embed_mask: (N, 1, seq_length)
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.

        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss') 
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            input from this dict will contain list of images, masks, embeddings and embedding masks.
                Shape after stack (all will be tensor)
                    images: (N, C, H, W) Ex: [2, 3, 360, 640]
                    masks: (N, 1, H, W)
                    embeddings: (N, 1, seq_length, feature dimension) (Bert: feature dim = 768) Ex: [2, 1, 300, 768]
                    embed_masks: (N, 1, seq_length)

        Returns:
            list: The predictions of given data.
        """

        data = self.data_preprocessor(data, False)
        # return self.predict(data['inputs']['img'], data['data_samples'], data['inputs']['mask'], data['inputs']['embedding'], data['inputs']['embedding_mask'])  # type: ignore
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            input from this dict will contain list of images, masks, embeddings and embedding masks.
                Shape after stack (all will be tensor)
                    images: (N, C, H, W) Ex: [2, 3, 360, 640]
                    masks: (N, 1, H, W)
                    embeddings: (N, 1, seq_length, feature dimension) (Bert: feature dim = 768) Ex: [2, 1, 300, 768]
                    embed_masks: (N, 1, seq_length)

        Returns:
            list: The predictions of given data.
        """

        data = self.data_preprocessor(data, False)
        # return self.predict(data['inputs']['img'], data['data_samples'], data['inputs']['mask'], data['inputs']['embedding'], data['inputs']['embedding_mask'])  # type: 
        return self._run_forward(data, mode='predict')  # type: ignore

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """

        images = inputs['img']
        masks = inputs['mask']
        embeddings = inputs['embedding']
        embed_masks = inputs['embedding_mask']

        if mode == 'loss':
            return self.loss(images, data_samples, masks, embeddings, embed_masks)
        elif mode == 'predict':
            return self.predict(images, data_samples, masks, embeddings, embed_masks) 
        elif mode == 'tensor':
            return self._forward(images, masks, embeddings, embed_masks, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def unnormalize(self, x):
        """
        Unnormalize images to return
        """
        mean = torch.tensor(self.data_preprocessor.mean).view(1, -1, 1, 1)
        std = torch.tensor(self.data_preprocessor.std).view(1, -1, 1, 1)
        return x * std + mean
    
    def normalize(self, x):
        """
        Normalize images
        """
        mean = torch.tensor(self.data_preprocessor.mean).view(1, -1, 1, 1)
        std = torch.tensor(self.data_preprocessor.std).view(1, -1, 1, 1)
        return (x - mean) / std

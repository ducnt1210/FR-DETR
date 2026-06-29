# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence, Dict
from tqdm import tqdm

import wandb
import random
import torch
import numpy as np
from torch.distributed import get_rank, is_initialized
from torch.nn.parallel import DistributedDataParallel as DDP

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import scale_boxes

import matplotlib.pyplot as plt  # For viridis colormap

def is_main_process():
    return not is_initialized() or get_rank() == 0

@HOOKS.register_module()
class DetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)


@HOOKS.register_module()
class DetWandbVisualizationHook(Hook):
    def __init__(self,
                 num_train_images: int = 5,
                 num_val_images: int = 5,
                 num_test_images: int = 10,
                 score_thr: float = 0.3,
                 backend_args: Optional[dict] = None):
        self._visualizer = Visualizer.get_current_instance()
        self.score_thr = score_thr
        self.backend_args = backend_args
        self.num_train_images = num_train_images
        self.num_val_images = num_val_images
        self.num_test_images = num_test_images
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        self.test_out_dir = "show_result"

    def before_train(self, runner: Runner) -> None:
        """Select image indices for logging during training."""
        dataset_length = len(runner.train_dataloader.dataset)
        self.train_indices = random.sample(range(dataset_length), min(self.num_train_images, dataset_length))

    def before_val(self, runner: Runner) -> None:
        """Select image indices for logging during validation."""
        dataset_length = len(runner.val_dataloader.dataset)
        self.val_indices = random.sample(range(dataset_length), min(self.num_val_images, dataset_length))

    def before_test(self, runner: Runner) -> None:
        """Select image indices for logging during testing."""
        dataset_length = len(runner.test_dataloader.dataset)
        self.test_indices = random.sample(range(dataset_length), min(self.num_test_images, dataset_length))

    def after_train_epoch(self, runner: Runner) -> None:
        """Log images after train epoch."""
        self._after_epoch(runner, mode='train')
        if is_initialized():
            if get_rank() == 0:
                self._log_images(runner, mode="train")
        else:
            self._log_images(runner, mode="train")

    # def before_train_epoch(self, runner) -> None:
    #     """All subclasses should override this method, if they need any
    #     operations before each training epoch.

    #     Args:
    #         runner (Runner): The runner of the training process.
    #     """
    #     self._before_epoch(runner, mode='train')
    #     if get_rank() == 0:
    #         print("This is rank", get_rank())
    #         self._log_images(runner, mode="train")

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Log images after validation epoch."""
        self._after_epoch(runner, mode='val')
        if is_initialized():
            if get_rank() == 0:
                self._log_images(runner, mode="val")
        else:
            self._log_images(runner, mode="val")
    
    def before_test_epoch(self, runner) -> None:
        self._before_epoch(runner, mode='test')
        if is_initialized():
            if get_rank() == 0:
                self._log_images_all_test(runner, mode="test")
                # self._log_4_extracted_feat_all_test(runner, mode="test")
        else:
            self._log_images_all_test(runner, mode="test")
            # self._log_4_extracted_feat_all_test(runner, mode="test")
        raise SystemExit

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """Log images after test epoch."""
        self._after_epoch(runner, mode='test')
        if is_initialized():
            if get_rank() == 0:
                self._log_images(runner, mode="test")
        else:
            self._log_images(runner, mode="test")

    def _log_images(self, runner: Runner, mode: str) -> None:
        """Helper function to log images to Wandb for train, val, or test."""
        indices = getattr(self, f"{mode}_indices")
        dataloader = getattr(runner, f"{mode}_dataloader")
        
        if mode == "train":
            runner.model.eval()  # Switch to evaluation mode for logging during training

        wandb_images = []
        for idx in indices:
            data = dataloader.dataset[idx]
            device = next(runner.model.parameters()).device

            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [to_device(v) for v in x]
                else:
                    return x

            batch_inputs = to_device(data['inputs'])
            batch_data_samples = [to_device(data['data_samples'])]

            # print(batch_data_samples[0].metainfo)  # Before modification
            if "batch_input_shape" not in batch_data_samples[0].metainfo:
                batch_data_samples[0].set_metainfo(dict(batch_input_shape = batch_data_samples[0].metainfo['img_shape']))
            else:
                scale_factor = batch_data_samples[0].metainfo['scale_factor']
                batch_data_samples[0].gt_instances.bboxes = scale_boxes(batch_data_samples[0].gt_instances.bboxes, scale_factor)
            # print(batch_data_samples[0].metainfo)  # After modification
            # print(batch_data_samples[0].metainfo['batch_input_shape'])

            # print("This is batch input", batch_inputs)
            # print("This is batch_data_samples", batch_data_samples)
            # raise SystemExit

            with torch.no_grad():
                if isinstance(runner.model, DDP):
                    model = runner.model.module
                else:
                    model = runner.model
                if isinstance(batch_inputs, dict):
                    batch_inputs = {k: v.unsqueeze(0) for k, v in batch_inputs.items()}
                    images = batch_inputs['img']
                    masks = batch_inputs['mask']
                    embeddings = batch_inputs['embedding']
                    embed_masks = batch_inputs['embedding_mask']
                    # The data first is load with opencv => It is BGR and the DetDataPreprocessor turn BGR->RGB for the model => Model run on RGB (We skip DetDataPreprocessor here)
                    images = images[:, [2, 1, 0], :, :]  # Swap the channels from BGR to RGB
                    enhanced_images, results = model.predict(images, batch_data_samples, masks, embeddings, embed_masks, rescale=False, return_enhanced_images=True, normalize_input=True)
    
                    img = batch_inputs['img'].squeeze(0).cpu().numpy()
                else:
                    batch_inputs = batch_inputs.unsqueeze(0)
                    batch_inputs = batch_inputs[:, [2, 1, 0], :, :]
                    enhanced_images, results = model.predict(batch_inputs, batch_data_samples=batch_data_samples, rescale=False, return_enhanced_images=True, normalize_input=True)
                      
                    img = batch_inputs.squeeze(0).cpu().numpy()

       #    img = batch_inputs['img'].squeeze(0).cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            img = mmcv.image.bgr2rgb(img)

            enhanced_images = enhanced_images.squeeze(0).cpu().numpy()
            enhanced_images = np.clip(enhanced_images, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            # enhanced_images = mmcv.image.bgr2rgb(enhanced_images) # Does not need to turn from bgr to rgb anymore cause the model now run on rgb images

            drawn_img = self._visualizer.visualize_datasample(
                img, results[0], pred_score_thr=self.score_thr)

            wandb_images.append(wandb.Image(np.hstack([drawn_img, enhanced_images]), 
                                            caption=f"Pre-enhanced (left) vs Enhanced (right). Image idx: {idx}"))
            


        wandb.log({f"{mode}_images": wandb_images}, step=runner.iter)

        if mode == "train":
            runner.model.train()  # Switch back to train mode after logging for training

    def _log_images_all_test(self, runner: Runner, mode: str) -> None:
        """Helper function to log images to Wandb for train, val, or test."""
        dataloader = getattr(runner, f"{mode}_dataloader")

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)
        
        if mode == "train":
            runner.model.eval()  # Switch to evaluation mode for logging during training

        for data in tqdm(dataloader.dataset):
            device = next(runner.model.parameters()).device

            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [to_device(v) for v in x]
                else:
                    return x

            batch_inputs = to_device(data['inputs'])
            batch_data_samples = [to_device(data['data_samples'])]
            img_path = data['data_samples'].img_path

            # print(batch_data_samples[0].metainfo)  # Before modification
            if "batch_input_shape" not in batch_data_samples[0].metainfo:
                batch_data_samples[0].set_metainfo(dict(batch_input_shape = batch_data_samples[0].metainfo['img_shape']))
            else:
                scale_factor = batch_data_samples[0].metainfo['scale_factor']
                batch_data_samples[0].gt_instances.bboxes = scale_boxes(batch_data_samples[0].gt_instances.bboxes, scale_factor)

            with torch.no_grad():
                if isinstance(runner.model, DDP):
                    model = runner.model.module
                else:
                    model = runner.model
                if isinstance(batch_inputs, dict):
                    batch_inputs = {k: v.unsqueeze(0) for k, v in batch_inputs.items()}
                    images = batch_inputs['img']
                    masks = batch_inputs['mask']
                    embeddings = batch_inputs['embedding']
                    embed_masks = batch_inputs['embedding_mask']
                    # The data first is load with opencv => It is BGR and the DetDataPreprocessor turn BGR->RGB for the model => Model run on RGB (We skip DetDataPreprocessor here)
                    images = images[:, [2, 1, 0], :, :]  # Swap the channels from BGR to RGB
                    enhanced_images, results = model.predict(images, batch_data_samples, masks, embeddings, embed_masks, rescale=False, return_enhanced_images=True, normalize_input=True)

                    img = batch_inputs['img'].squeeze(0).cpu().numpy()
                else:
                    batch_inputs = batch_inputs.unsqueeze(0)
                    batch_inputs = batch_inputs[:, [2, 1, 0], :, :]
                    enhanced_images, results = model.predict(batch_inputs, batch_data_samples=batch_data_samples, rescale=False, return_enhanced_images=True, normalize_input=True)

                    img = batch_inputs.squeeze(0).cpu().numpy()

            # img = batch_inputs['img'].squeeze(0).cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            img = mmcv.image.bgr2rgb(img)

            enhanced_images = enhanced_images.squeeze(0).cpu().numpy()
            enhanced_images = np.clip(enhanced_images, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            # enhanced_images = mmcv.image.bgr2rgb(enhanced_images) # Does not need to turn from bgr to rgb anymore cause the model now run on rgb images

            out_file = osp.basename(img_path)
            out_file = osp.join(self.test_out_dir, out_file)

            drawn_img = self._visualizer.visualize_datasample_only_enhanced(
                enhanced_images, results[0], pred_score_thr=self.score_thr)
            
            if out_file is not None:
                # mmcv.imwrite(enhanced_images[..., ::-1], out_file)
                mmcv.imwrite(drawn_img[..., ::-1], out_file)
            

    def _log_4_extracted_feat_all_test(self, runner: Runner, mode: str) -> None:
        dataloader = getattr(runner, f"{mode}_dataloader")

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)
        
        if mode == "train":
            runner.model.eval()  # Switch to evaluation mode for logging during training

        for data in tqdm(dataloader.dataset):
            device = next(runner.model.parameters()).device

            def to_device(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device)
                elif isinstance(x, dict):
                    return {k: to_device(v) for k, v in x.items()}
                elif isinstance(x, list):
                    return [to_device(v) for v in x]
                else:
                    return x

            batch_inputs = to_device(data['inputs'])
            batch_data_samples = [to_device(data['data_samples'])]
            img_path = data['data_samples'].img_path

            # print(batch_data_samples[0].metainfo)  # Before modification
            if "batch_input_shape" not in batch_data_samples[0].metainfo:
                batch_data_samples[0].set_metainfo(dict(batch_input_shape = batch_data_samples[0].metainfo['img_shape']))
            else:
                scale_factor = batch_data_samples[0].metainfo['scale_factor']
                batch_data_samples[0].gt_instances.bboxes = scale_boxes(batch_data_samples[0].gt_instances.bboxes, scale_factor)

            with torch.no_grad():
                if isinstance(runner.model, DDP):
                    model = runner.model.module
                else:
                    model = runner.model
                if isinstance(batch_inputs, dict):
                    batch_inputs = {k: v.unsqueeze(0) for k, v in batch_inputs.items()}
                    images = batch_inputs['img']
                    masks = batch_inputs['mask']
                    embeddings = batch_inputs['embedding']
                    embed_masks = batch_inputs['embedding_mask']
                    # The data first is load with opencv => It is BGR and the DetDataPreprocessor turn BGR->RGB for the model => Model run on RGB (We skip DetDataPreprocessor here)
                    images = images[:, [2, 1, 0], :, :]  # Swap the channels from BGR to RGB
                    enhanced_images, results = model.predict(images, batch_data_samples, masks, embeddings, embed_masks, rescale=False, return_enhanced_images=True, normalize_input=True)

                    img = batch_inputs['img'].squeeze(0).cpu().numpy()
                else:
                    batch_inputs = batch_inputs.unsqueeze(0)
                    batch_inputs = batch_inputs[:, [2, 1, 0], :, :]
                    enhanced_images, results = model.predict(batch_inputs, batch_data_samples=batch_data_samples, rescale=False, return_enhanced_images=True, normalize_input=True)

                    img = batch_inputs.squeeze(0).cpu().numpy()

            # img = batch_inputs['img'].squeeze(0).cpu().numpy()
            img = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 2, 0)
            img = mmcv.image.bgr2rgb(img)

            # enhanced_images has shape (B, 4, H, W). Process each channel.
            enhanced_images = enhanced_images.squeeze(0).cpu().numpy()  # Shape (4, H, W)

            base_name = osp.basename(img_path)
            base_name, ext = osp.splitext(base_name)

            for i in range(4):
                feature_map = enhanced_images[i]  # Shape (H, W)

                # Normalize the feature map to [0, 1]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)

                # Save the feature map using Matplotlib to ensure correct colormap application
                out_file = osp.join(self.test_out_dir, f"{base_name}_feature_{i + 1}.png")
                
                plt.imsave(out_file, feature_map, cmap='viridis')



@HOOKS.register_module()
class TrackVisualizationHook(Hook):
    """Tracking Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        frame_interval (int): The interval of visualization. Defaults to 30.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict): Arguments to instantiate a file client.
            Defaults to ``None``.
    """

    def __init__(self,
                 draw: bool = False,
                 frame_interval: int = 30,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None) -> None:
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.frame_interval = frame_interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self.image_idx = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[TrackDataSample]) -> None:
        """Run after every ``self.interval`` validation iteration.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1,\
            'only batch_size=1 is supported while validating.'

        sampler = runner.val_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                total_curr_iter = runner.iter + batch_idx
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0],
                                            total_curr_iter)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        total_curr_iter = runner.iter + self.image_idx + \
                                          frame_id
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    total_curr_iter)
                self.image_idx = self.image_idx + video_length

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[TrackDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        sampler = runner.test_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0], batch_idx)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    self.image_idx + frame_id)
                self.image_idx = self.image_idx + video_length

    def visualize_single_image(self, img_data_sample: DetDataSample,
                               step: int) -> None:
        """
        Args:
            img_data_sample (DetDataSample): single image output.
            step (int): The index of the current image.
        """
        img_path = img_data_sample.img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        out_file = None
        if self.test_out_dir is not None:
            video_name = img_path.split('/')[-3]
            mkdir_or_exist(osp.join(self.test_out_dir, video_name))
            out_file = osp.join(self.test_out_dir, video_name,
                                osp.basename(img_path))

        self._visualizer.add_datasample(
            osp.basename(img_path) if self.show else 'test_img',
            img,
            data_sample=img_data_sample,
            show=self.show,
            wait_time=self.wait_time,
            pred_score_thr=self.score_thr,
            out_file=out_file,
            step=step)

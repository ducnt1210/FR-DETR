# save as: tools/check_dataset_samples.py
import argparse
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules


def check_batch(batch, num_classes: int, batch_idx: int) -> int:
    """Return count of anomalies found."""
    num_issues = 0

    # inputs: List[Tensor] - list of image tensors [C, H, W]
    inputs = batch['inputs']  # List[torch.Tensor]
    data_samples = batch['data_samples']  # list of DetDataSample

    # 1) Inputs sanity - check each image tensor in the batch
    for i, img_tensor in enumerate(inputs):
        if not torch.isfinite(img_tensor).all():
            finite = torch.isfinite(img_tensor)
            imin = img_tensor[finite].min().item() if finite.any() else float('nan')
            imax = img_tensor[finite].max().item() if finite.any() else float('nan')
            print(f'[Batch {batch_idx} Image {i}] Non-finite image tensor: min={imin}, max={imax}')
            num_issues += 1

    # 2) Per-sample GT sanity
    for i, sample in enumerate(data_samples):
        img_shape = sample.metainfo.get('img_shape', None)
        if img_shape is not None:
            h, w = img_shape
            if not (isinstance(h, (int, float)) and isinstance(w, (int, float)) and h > 0 and w > 0):
                print(f'[Batch {batch_idx} Sample {i}] Bad img_shape: {img_shape}')
                num_issues += 1

        gt = sample.gt_instances
        if hasattr(gt, 'bboxes'):
            bboxes = gt.bboxes  # HorizontalBoxes object
            if hasattr(bboxes, 'tensor'):
                bbox_tensor = bboxes.tensor  # (K, 4) xyxy tensor
            else:
                bbox_tensor = bboxes  # fallback for direct tensor
            
            if bbox_tensor.numel() > 0:
                if not torch.isfinite(bbox_tensor).all():
                    finite = torch.isfinite(bbox_tensor)
                    bmin = bbox_tensor[finite].min().item() if finite.any() else float('nan')
                    bmax = bbox_tensor[finite].max().item() if finite.any() else float('nan')
                    print(f'[Batch {batch_idx} Sample {i}] Non-finite GT bboxes: min={bmin}, max={bmax}')
                    num_issues += 1

                # Degenerate boxes: x2<x1 or y2<y1 or zero/negative size
                x1, y1, x2, y2 = bbox_tensor[:, 0], bbox_tensor[:, 1], bbox_tensor[:, 2], bbox_tensor[:, 3]
                bad_order = (x2 < x1) | (y2 < y1)
                zero_w = (x2 - x1) <= 0
                zero_h = (y2 - y1) <= 0
                degenerate = bad_order | zero_w | zero_h
                if degenerate.any():
                    idxs = torch.nonzero(degenerate, as_tuple=False).squeeze(1).tolist()
                    print(f'[Batch {batch_idx} Sample {i}] Degenerate GT boxes at indices: {idxs}')
                    num_issues += 1

        if hasattr(gt, 'labels'):
            labels = gt.labels  # (K,)
            if labels.numel() > 0:
                if not torch.isfinite(labels.float()).all():
                    print(f'[Batch {batch_idx} Sample {i}] Non-finite GT labels detected')
                    num_issues += 1
                min_label = int(labels.min().item())
                max_label = int(labels.max().item())
                if min_label < 0 or max_label >= num_classes:
                    print(f'[Batch {batch_idx} Sample {i}] Out-of-range labels: min={min_label}, max={max_label}, '
                          f'num_classes={num_classes}')
                    num_issues += 1

    return num_issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Path to config.py')
    parser.add_argument('--loader', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--num-batches', type=int, default=10)
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg)
    # Ensure MMDetection registries (datasets, samplers, transforms) are registered
    register_all_modules(init_default_scope=True)

    # Infer num_classes from model head or dataset metainfo if provided
    num_classes = None
    try:
        num_classes = cfg.model['bbox_head']['num_classes']
    except Exception:
        pass
    # Fallback to dataset metainfo classes if available
    try:
        if args.loader == 'train':
            classes = cfg.train_dataloader['dataset'].get('metainfo', {}).get('classes', None)
        elif args.loader == 'val':
            classes = cfg.val_dataloader['dataset'].get('metainfo', {}).get('classes', None)
        else:
            classes = cfg.test_dataloader['dataset'].get('metainfo', {}).get('classes', None)
        if num_classes is None and classes is not None:
            num_classes = len(classes)
    except Exception:
        pass

    if num_classes is None:
        print('[WARN] Could not determine num_classes from cfg. Defaulting to 80.')
        num_classes = 80

    # Build dataset + dataloader
    if args.loader == 'train':
        dl_cfg = cfg.train_dataloader
    elif args.loader == 'val':
        dl_cfg = cfg.val_dataloader
    else:
        dl_cfg = cfg.test_dataloader

    dataset = DATASETS.build(dl_cfg['dataset'])
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    dl_cfg = dict(dl_cfg)  # shallow copy
    dl_cfg['dataset'] = dataset
    data_loader = Runner.build_dataloader(dl_cfg)

    print(f'Checking first {args.num_batches} batches from {args.loader} loader...')
    total_issues = 0
    for batch_idx, batch in enumerate(data_loader):
        total_issues += check_batch(batch, num_classes, batch_idx)
        if batch_idx + 1 >= args.num_batches:
            break

    if total_issues == 0:
        print('No anomalies found in the inspected batches.')
    else:
        print(f'Found {total_issues} anomalies in the inspected batches.')


if __name__ == '__main__':
    main()
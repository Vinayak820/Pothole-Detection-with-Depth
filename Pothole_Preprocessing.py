"""
Pothole Dataset EDA and Preprocessing
-------------------------------------
Usage:
    python pothole_dataset_eda_preprocess.py \
        --dataset_dir "/path/to/PUBLIC POTHHOLE DATASET" \
        --output_dir "./processed" \
        --target_size 640

What this script does:
  - Scans the dataset directory (expects `images/`, `depths/`, `labels/` subfolders)
  - Builds a DataFrame pairing image, depth (.npy) and label (.txt) files using the timestamp
    prefix like `20250227_135438` found in filenames.
  - Performs EDA: counts, shapes, depth statistics, annotation counts, mask area distribution,
    and plots (histograms, scatter plots). Saves plots to output_dir/eda_plots.
  - Provides visualization utilities to overlay segmentation polygons / masks on RGB and
    show depth maps.
  - Preprocessing pipeline: resize images & depth to `target_size`, normalize depth optionally,
    convert polygon labels to normalized YOLOv8 polygon format (if needed) and write a manifest CSV.

Notes on label parsing:
  - The code tries to support common Roboflow / YOLO text formats:
      * YOLO bbox format (class x_center y_center width height) -- 5 numbers per line
      * Segmentation polygon format (class x1 y1 x2 y2 ...) -- >5 numbers per line
      * Comma separated polygon coords (x1,y1,x2,y2,...)
      * JSON encoded lists
  - Coordinates are auto-detected as normalized (0..1) or absolute pixel coords.

Dependencies:
  - numpy, pandas, matplotlib, opencv-python, pillow, tqdm

Make sure to `pip install numpy pandas matplotlib opencv-python pillow tqdm` before running.
"""

import os
import re
import argparse
import glob
import json
import ast
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm


TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def find_timestamp(name: str):
    m = TIMESTAMP_RE.search(name)
    if m:
        return m.group(1)
    return None


def gather_files(dataset_dir):
    images_dir = os.path.join(dataset_dir, 'images')
    depths_dir = os.path.join(dataset_dir, 'depths')
    labels_dir = os.path.join(dataset_dir, 'labels')

    image_files = sorted(glob.glob(os.path.join(images_dir, '*')))
    depth_files = sorted(glob.glob(os.path.join(depths_dir, '*.npy')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*')))

    # map timestamp -> file paths
    idx = {}

    for p in image_files:
        ts = find_timestamp(os.path.basename(p))
        if not ts:
            continue
        idx.setdefault(ts, {})['image'] = p

    for p in depth_files:
        ts = find_timestamp(os.path.basename(p))
        if not ts:
            continue
        idx.setdefault(ts, {})['depth'] = p

    for p in label_files:
        ts = find_timestamp(os.path.basename(p))
        if not ts:
            continue
        # Keep list of label files per timestamp (sometimes multiple annotations)
        idx.setdefault(ts, {}).setdefault('labels', []).append(p)

    # Build dataframe rows
    rows = []
    for ts, d in sorted(idx.items()):
        rows.append({
            'timestamp': ts,
            'image': d.get('image'),
            'depth': d.get('depth'),
            'labels': d.get('labels', [])
        })

    df = pd.DataFrame(rows)
    return df


# label parsing

def try_parse_line(line):
    """Try different parsing strategies for a single label line."""
    line = line.strip()
    if not line:
        return None
    # common separators: whitespace or comma
    # try JSON first
    try:
        obj = json.loads(line)
        return obj
    except Exception:
        pass
    # try python literal
    try:
        obj = ast.literal_eval(line)
        return obj
    except Exception:
        pass
    # finally fall back to whitespace or comma separated floats
    if ',' in line and ' ' not in line:
        parts = line.split(',')
    else:
        parts = line.split()
    try:
        nums = [float(p) for p in parts]
        return nums
    except Exception:
        # give up
        return line


def parse_label_file(label_path, img_w, img_h):
    """Parse a label file into a list of annotation dicts.
    Each annotation: {'class': int, 'type': 'bbox'|'poly', 'coords': [...] }
    coords for bbox: [x_center_norm, y_center_norm, w_norm, h_norm] or pixel coords if >1
    coords for poly: [x1,y1,x2,y2,...] normalized or absolute depending on values
    """
    anns = []
    with open(label_path, 'r') as f:
        for line in f:
            parsed = try_parse_line(line)
            if parsed is None:
                continue
            # parsed can be list, dict, or string
            if isinstance(parsed, dict):
                # Roboflow JSON-style: expect 'annotations' or 'class' etc — try to extract useful bits
                # We keep JSON as-is for now
                anns.append({'class': parsed.get('class', 0), 'type': 'json', 'raw': parsed})
                continue
            if isinstance(parsed, (list, tuple)):
                nums = list(parsed)
                # if first element is integer class id and rest coords
                if len(nums) >= 5 and float(nums[0]).is_integer():
                    cls = int(nums[0])
                    coords = nums[1:]
                else:
                    # maybe no class provided, assume class 0
                    cls = 0
                    coords = nums
                if len(coords) == 4:
                    # treat as bbox
                    anns.append({'class': cls, 'type': 'bbox', 'coords': coords})
                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    # polygon
                    anns.append({'class': cls, 'type': 'poly', 'coords': coords})
                else:
                    # unknown layout
                    anns.append({'class': cls, 'type': 'unknown', 'coords': coords})
                continue
            # fallback: line as string
            s = str(parsed)
            # try whitespace-separated numbers
            parts = s.replace(',', ' ').split()
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 4:
                    anns.append({'class': 0, 'type': 'bbox', 'coords': nums})
                elif len(nums) >= 6 and len(nums) % 2 == 0:
                    anns.append({'class': 0, 'type': 'poly', 'coords': nums})
                else:
                    anns.append({'class': 0, 'type': 'unknown', 'coords': nums})
            except Exception:
                # unparseable
                anns.append({'class': 0, 'type': 'text', 'raw': s})
    return anns


def poly_coords_to_mask(poly_coords, img_w, img_h):
    # poly_coords: [x1,y1,x2,y2,...] either normalized (0..1) or absolute (pixel)
    coords = np.array(poly_coords).reshape(-1, 2)
    # detect normalized
    if coords.max() <= 1.0001:
        coords[:, 0] = coords[:, 0] * img_w
        coords[:, 1] = coords[:, 1] * img_h
    else:
        # assume absolute pixels
        pass
    pts = coords.astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def bbox_to_mask(bbox_coords, img_w, img_h):
    # bbox_coords: [x_center, y_center, w, h] may be normalized (0..1) or absolute
    x_center, y_center, w, h = bbox_coords
    if max(x_center, y_center, w, h) <= 1.0001:
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h
    x1 = int(round(x_center - w / 2.0))
    y1 = int(round(y_center - h / 2.0))
    x2 = int(round(x_center + w / 2.0))
    y2 = int(round(y_center + h / 2.0))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w-1, x2); y2 = min(img_h-1, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2+1, x1:x2+1] = 1
    return mask


def visualize_overlay(img, mask, alpha=0.5, title=None):
    """Show image with mask overlay (red)."""
    img_rgb = img.copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(img_rgb)
    color_mask[:, :, 2] = (mask * 255).astype(np.uint8)
    overlayed = cv2.addWeighted(img_rgb, 1.0, color_mask, alpha, 0)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    plt.show()


def show_depth(depth_arr, title=None):
    plt.figure(figsize=(6,4))
    plt.axis('off')
    d = depth_arr.copy()
    # clamp and normalize for visualization
    d = np.nan_to_num(d, nan=0.0)
    vmin = np.percentile(d, 2)
    vmax = np.percentile(d, 98)
    plt.imshow(d, cmap='viridis', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.show()


def perform_eda(df, output_dir, sample_n=6):
    os.makedirs(output_dir, exist_ok=True)
    stats = {}
    # basic counts
    stats['total_samples'] = len(df)
    stats['with_image'] = df['image'].notnull().sum()
    stats['with_depth'] = df['depth'].notnull().sum()
    stats['with_labels'] = df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

    print('Dataset counts:', stats)

    # collect more detailed metrics
    depths_min = []
    depths_max = []
    depths_mean = []
    ann_counts = []
    mask_area_fracs = []
    mean_depth_in_masks = []

    # sample a subset for speed during EDA if dataset is large
    rows = df.to_dict('records')
    for r in tqdm(rows, desc='EDA loop'):
        if r['image'] is None or r['depth'] is None:
            ann_counts.append(0)
            continue
        try:
            # load image for shape only
            img = Image.open(r['image'])
            img_w, img_h = img.size
            # load depth
            depth = np.load(r['depth'])
            if depth.shape[0] != img_h or depth.shape[1] != img_w:
                # depth could be transposed or smaller; try to handle gracefully
                depth = cv2.resize(depth.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            dmin = float(np.nanmin(depth))
            dmax = float(np.nanmax(depth))
            dmean = float(np.nanmean(np.nan_to_num(depth)))
            depths_min.append(dmin); depths_max.append(dmax); depths_mean.append(dmean)

            # parse labels and compute mask area & depth in mask
            total_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            ann_list = []
            for lab in r['labels']:
                anns = parse_label_file(lab, img_w, img_h)
                ann_list.extend(anns)
            ann_counts.append(len(ann_list))
            for ann in ann_list:
                if ann['type'] == 'poly':
                    mask = poly_coords_to_mask(ann['coords'], img_w, img_h)
                    total_mask = np.clip(total_mask + mask, 0, 1)
                elif ann['type'] == 'bbox':
                    mask = bbox_to_mask(ann['coords'], img_w, img_h)
                    total_mask = np.clip(total_mask + mask, 0, 1)
            area_frac = float(total_mask.sum()) / (img_w * img_h)
            mask_area_fracs.append(area_frac)
            if total_mask.sum() > 0:
                mean_d_mask = float(np.nanmean(depth[total_mask.astype(bool)]))
                mean_depth_in_masks.append(mean_d_mask)
            else:
                mean_depth_in_masks.append(np.nan)
        except Exception as e:
            # if any sample fails, skip but log
            # print('EDA sample error:', e)
            ann_counts.append(0)
            depths_min.append(np.nan); depths_max.append(np.nan); depths_mean.append(np.nan)
            mask_area_fracs.append(np.nan); mean_depth_in_masks.append(np.nan)

    eda_df = pd.DataFrame({
        'depth_min': depths_min,
        'depth_max': depths_max,
        'depth_mean': depths_mean,
        'ann_count': ann_counts,
        'mask_area_frac': mask_area_fracs,
        'mean_depth_in_mask': mean_depth_in_masks
    })

    # save summary
    eda_df.to_csv(os.path.join(output_dir, 'eda_sample_stats.csv'), index=False)

    # plots
    plt.figure(figsize=(6,4))
    plt.hist([v for v in depths_mean if not np.isnan(v)], bins=50)
    plt.title('Histogram of image mean depths')
    plt.xlabel('depth (units as in .npy)')
    plt.savefig(os.path.join(output_dir, 'hist_depth_mean.png'))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist([v for v in mask_area_fracs if not np.isnan(v)], bins=40)
    plt.title('Mask area fraction distribution')
    plt.xlabel('area fraction')
    plt.savefig(os.path.join(output_dir, 'hist_mask_area_frac.png'))
    plt.close()


#------------------------------------
    # plt.figure(figsize=(6,5))
    # plt.scatter([a for a in mask_area_fracs if not np.isnan(a)], [d for d in mean_depth_in_masks if not np.isnan(d)], s=12)
    # plt.xlabel('mask_area_frac'); plt.ylabel('mean_depth_in_mask'); plt.title('Mask area vs mean depth')
    # plt.savefig(os.path.join(output_dir, 'mask_area_vs_mean_depth.png'))
    # plt.close()
    plt.figure(figsize=(6,5))

    # Filter both lists together so sizes remain equal
    valid_x = []
    valid_y = []

    for a, d in zip(mask_area_fracs, mean_depth_in_masks):
        if not np.isnan(a) and not np.isnan(d):
            valid_x.append(a)
            valid_y.append(d)

    if len(valid_x) > 0:
        plt.scatter(valid_x, valid_y, s=12)
        plt.xlabel('mask_area_frac')
        plt.ylabel('mean_depth_in_mask')
        plt.title('Mask area vs mean depth')
        plt.savefig(os.path.join(output_dir, 'mask_area_vs_mean_depth.png'))

    plt.close()
#-------------------------------------------------

    print('EDA plots saved to', output_dir)
    return eda_df


def preprocess_and_save(df, output_dir, target_size=640, normalize_depth=True):
    os.makedirs(output_dir, exist_ok=True)
    od_img = os.path.join(output_dir, 'images'); os.makedirs(od_img, exist_ok=True)
    od_depth = os.path.join(output_dir, 'depths'); os.makedirs(od_depth, exist_ok=True)
    od_labels = os.path.join(output_dir, 'labels'); os.makedirs(od_labels, exist_ok=True)

    manifest_rows = []

    for r in tqdm(df.to_dict('records'), desc='Preprocess'):
        ts = r['timestamp']
        if r['image'] is None or r['depth'] is None:
            continue
        try:
            img = cv2.imread(r['image'])
            img_h, img_w = img.shape[:2]
            depth = np.load(r['depth']).astype(np.float32)
        except Exception as e:
            print('skip', r['image'], 'error', e)
            continue
        # resize
        target_h = target_w = target_size
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # depth resize: preserve numeric values, use NEAREST or LINEAR
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if normalize_depth:
            # min-max normalize per-image to 0..1 for model input; keep copy of raw depth
            dmin = np.nanmin(depth_resized); dmax = np.nanmax(depth_resized)
            if dmax - dmin > 1e-6:
                depth_norm = (depth_resized - dmin) / (dmax - dmin)
            else:
                depth_norm = depth_resized - dmin
        else:
            depth_norm = depth_resized

        # write outputs
        out_img_p = os.path.join(od_img, f'{ts}_img.jpg')
        out_depth_p = os.path.join(od_depth, f'{ts}_depth.npy')
        cv2.imwrite(out_img_p, img_resized)
        np.save(out_depth_p, depth_norm)

        # convert labels to normalized polygons for YOLOv8-seg if possible
        out_label_p = os.path.join(od_labels, f'{ts}.txt')
        all_anns = []
        if r['labels']:
            for lab in r['labels']:
                anns = parse_label_file(lab, img_w, img_h)
                for ann in anns:
                    if ann['type'] == 'poly':
                        coords = np.array(ann['coords']).reshape(-1,2)
                        # convert absolute -> normalized
                        if coords.max() > 1.0:
                            coords[:,0] = coords[:,0] / img_w
                            coords[:,1] = coords[:,1] / img_h
                        flat = coords.ravel().tolist()
                        # YOLOv8 seg format: class x1 y1 x2 y2 ... (normalized)
                        line = str(int(ann.get('class',0))) + ' ' + ' '.join([f'{v:.6f}' for v in flat])
                        all_anns.append(line)
                    elif ann['type'] == 'bbox':
                        # convert bbox to polygon (optional) or keep bbox
                        bx = ann['coords']
                        # if bbox normalized, keep as bbox (YOLO bbox 5 numbers)
                        if max(bx) <= 1.0:
                            line = str(int(ann.get('class',0))) + ' ' + ' '.join([f'{v:.6f}' for v in bx])
                        else:
                            # convert to normalized
                            x_c = bx[0] / img_w; y_c = bx[1] / img_h; w_ = bx[2] / img_w; h_ = bx[3] / img_h
                            line = str(int(ann.get('class',0))) + ' ' + ' '.join([f'{v:.6f}' for v in [x_c,y_c,w_,h_]])
                        all_anns.append(line)
                    else:
                        # unknown or json - skip or save raw
                        pass
        # write label file
        with open(out_label_p, 'w') as f:
            for l in all_anns:
                f.write(l + '\n')

        manifest_rows.append({
            'timestamp': ts,
            'image': out_img_p,
            'depth': out_depth_p,
            'label': out_label_p,
            'orig_image': r['image'],
            'orig_depth': r['depth'],
            'num_annots': len(all_anns)
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(output_dir, 'manifest.csv'), index=False)
    print('Preprocessing finished. Manifest saved to', os.path.join(output_dir, 'manifest.csv'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset root (contains images/, depths/, labels/)')
    parser.add_argument('--output_dir', type=str, default='./processed', help='Where to save EDA outputs and processed files')
    parser.add_argument('--target_size', type=int, default=640, help='Square target size to resize images/depth')
    parser.add_argument('--skip_eda', action='store_true')
    parser.add_argument('--normalize_depth', action='store_true')
    args = parser.parse_args()

    ds = args.dataset_dir
    out = args.output_dir
    print('Scanning dataset at', ds)
    df = gather_files(ds)
    print('Found', len(df), 'timestamp groups')
    print(df.head())

    eda_out = os.path.join(out, 'eda_plots')
    if not args.skip_eda:
        eda_df = perform_eda(df, eda_out)

    manifest = preprocess_and_save(df, os.path.join(out, 'preprocessed'), target_size=args.target_size, normalize_depth=args.normalize_depth)

    print('Done.')

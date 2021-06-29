"""Raw2raw data: raw extraction and paired set processing

If you use our dataset or this code, please cite the following paper:

M. Afifi and A. Abuolaim. Semi-Supervised Raw-to-Raw Mapping. arXiv preprint
2021.

@article{afifi2021raw2raw,
  title={Semi-Supervised Raw-to-Raw Mapping},
  author={Afifi, Mahmoud and Abuolaim, Abdullah},
  journal={arXiv preprint arXiv:2106.13883},
  year={2021}
}
Code written by: Abdullah Abuolaim
"""

from scipy.io import loadmat
import cv2
import numpy as np
import rawpy
from scipy.io import savemat
import os
import errno
from copy import deepcopy


def check_dir(_path):
  if not os.path.exists(_path):
    try:
      os.makedirs(_path)
    except OSError as exc:  # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise


def pack_rggb(raw_image, _cfa):
  height, width = raw_image.shape
  channels = []
  _cfa[_cfa == 2] += 1
  _cfa[2:][_cfa[2:] == 1] += 1
  idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
  for c in _cfa:
    raw_image_c = raw_image[idx[c][0]:height:2, idx[c][1]:width:2].copy()
    channels.append(raw_image_c)
  channels = np.stack(channels, axis=-1)
  return channels


def imwrite(filename, image):
  image = image * 256
  image = image.astype(np.uint8)
  image = from_rgb2bgr(image)
  cv2.imwrite(filename, image)


def from_rgb2bgr(im):
  return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def kernelP(rggb):
  r, gr, gb, b = np.split(rggb, 4, axis=1)
  return np.concatenate([rggb, rggb ** 2, r * gr, r * gb, r * b, gr * gb,
                         gr * b, gb * b, r * gr * gb * b, np.ones_like(r)],
                        axis=1)


def mapping(img, matrix):
  h, w, c = img.shape
  img = np.reshape(img, (-1, 4))
  mapped = kernelP(img) @ matrix
  mapped = np.reshape(mapped, (h, w, c))
  return mapped


def from_rggb_to_rgb(rggb):
  g = (rggb[:, :, 1] + rggb[:, :, 2]) / 2
  rggb[:, :, 1] = g
  rggb[:, :, 2] = rggb[:, :, 3]
  return rggb[:, :, :3]


pair_data = ['paired/', 'unpaired/']
cameras = ['iphone-x/', 'samsung-s9/']

meta_data_samsung = {'white_level': (2 ** 10) - 1, 'black_level': 0,
                     'cfa_pattern': np.array([1, 0, 2, 1])}
meta_data_iphone = {'white_level': (2 ** 12) - 1, 'black_level': 528,
                    'cfa_pattern': np.array([0, 1, 1, 2])}

anchor_files = [['2021-06-03_20-43-09.dng', '2021-06-03_20-48-56.dng',
                 '2021-06-03_21-11-57.dng', '2021-06-03_21-17-18.dng',
                 '2021-06-03_21-32-38.dng', '2021-06-03_22-03-52.dng',
                 '2021-06-07_16-13-49.dng', '2021-06-07_16-18-15.dng',
                 '2021-06-07_16-27-27.dng', '2021-06-07_16-31-10.dng',
                 '2021-06-07_16-34-42.dng', '2021-06-07_16-39-03.dng'],
                ['2021-06-03_20-43-08.dng', '2021-06-03_20-48-55.dng',
                 '2021-06-03_21-11-55.dng', '2021-06-03_21-17-16.dng',
                 '2021-06-03_21-32-39.dng', '2021-06-03_22-00-28.dng',
                 '2021-06-07_16-13-48.dng', '2021-06-07_16-18-14.dng',
                 '2021-06-07_16-27-25.dng', '2021-06-07_16-31-14.dng',
                 '2021-06-07_16-34-41.dng', '2021-06-07_16-39-03.dng']
                ]

samsung_white_level = 1023
samsung_black_level = 0
samsung_cfa_pattern = [1, 0, 2, 1]


for _pair in pair_data:
  for _cam in cameras:
    if _cam == cameras[0]:
      postfix = '_A'
      if _pair == 'paired/':
        a_files = anchor_files[0]
    else:
      postfix = '_B'
      if _pair == 'paired/':
        a_files = anchor_files[1]

    '''set the path to the RAW data'''
    path_to_raw_data = './dataset/' + _pair + _cam

    write_raw_rggb_path = path_to_raw_data + 'raw-rggb/'
    write_raw_vis_path = path_to_raw_data + 'vis/'
    if _pair == 'paired/':
      write_anchor_raw_rggb_path = path_to_raw_data + 'anchor-raw-rggb/'
      write_anchor_raw_vis_path = path_to_raw_data + 'anchor-vis/'
      check_dir(write_anchor_raw_rggb_path)
      check_dir(write_anchor_raw_vis_path)

    if _pair == 'paired/':
      mapping_path = path_to_raw_data + 'mapping/'
      write_raw_rggb_path_mapping = './dataset/' + _pair + list(
        set(cameras) - {_cam})[0] + 'raw-rggb/'
      write_raw_vis_path_mapping = './dataset/' + _pair + list(
        set(cameras) - {_cam})[0] + 'vis/'
      write_anchor_raw_rggb_path_mapping = './dataset/' + _pair + list(
        set(cameras) - {_cam})[0] + 'anchor-raw-rggb/'
      write_anchor_raw_vis_path_mapping = './dataset/' + _pair + list(
        set(cameras) - {_cam})[0] + 'anchor-vis/'
      check_dir(write_raw_rggb_path_mapping)
      check_dir(write_raw_vis_path_mapping)
      check_dir(write_anchor_raw_rggb_path_mapping)
      check_dir(write_anchor_raw_vis_path_mapping)


    else:
      mapping_path = None

    check_dir(write_raw_rggb_path)
    check_dir(write_raw_vis_path)

    all_raw_img_paths = [path_to_raw_data + '/dng/' + f for f in
                         os.listdir(path_to_raw_data + '/dng/') if
                         f.endswith(('.DNG', '.dng'))]
    all_raw_img_paths.sort()

    if _cam == 'samsung-s9/':
      meta_data = meta_data_samsung
    else:
      meta_data = meta_data_iphone

    for raw_img_path in all_raw_img_paths:
      raw_bayer = rawpy.imread(raw_img_path).raw_image_visible.copy()

      if _cam == 'samsung-s9/':
        raw_bayer_norm = (raw_bayer - meta_data['black_level']) / (
              meta_data['white_level'] - meta_data['black_level'])
      else:
        raw_bayer_norm = (raw_bayer.astype(np.float32) - meta_data[
          'black_level']) / (meta_data['white_level'] - meta_data[
          'black_level'])

      raw_bayer_norm[raw_bayer_norm < 0] = 0
      raw_bayer_norm[raw_bayer_norm > 1] = 1

      # pack (stack rggb) based on sensor pattern
      raw_rggb_chs = pack_rggb(raw_bayer_norm,
                               deepcopy(meta_data['cfa_pattern']))

      temp_img_name = (raw_img_path.split('/')[-1]).split('.')[0]

      '''mapping'''
      if mapping_path is not None:
        mapping_matrix = loadmat(mapping_path + temp_img_name + '.mat')
        mapping_matrix = mapping_matrix['mapping_matrix']

        mapped_raw_rggb_chs = mapping(raw_rggb_chs, mapping_matrix)
        mapped_raw_rggb_chs[mapped_raw_rggb_chs < 0] = 0
        mapped_raw_rggb_chs[mapped_raw_rggb_chs > 1] = 1

      '''
      Save data .mat
      '''
      # save metadata
      if _pair == 'paired/':
        if temp_img_name + '.dng' in a_files:
          raw_rggb_path = write_anchor_raw_rggb_path
          vis_path = write_anchor_raw_vis_path
          raw_rggb_mapping_path = write_anchor_raw_rggb_path_mapping
          vis_mapping_path = write_anchor_raw_vis_path_mapping
        else:
          raw_rggb_path = write_raw_rggb_path
          vis_path = write_raw_vis_path
          raw_rggb_mapping_path = write_raw_rggb_path_mapping
          vis_mapping_path = write_raw_vis_path_mapping

      else:
        raw_rggb_path = write_raw_rggb_path
        vis_path = write_raw_vis_path

      # save raw image
      raw_rggb_chs_mat = {"raw_rggb": raw_rggb_chs}
      savemat(raw_rggb_path + temp_img_name + postfix + '.mat',
              raw_rggb_chs_mat)

      # save mapped image (if paired)
      if mapping_path is not None:
        mapped_raw_rggb_chs_mat = {"raw_rggb": mapped_raw_rggb_chs}
        savemat(raw_rggb_mapping_path + temp_img_name + postfix + '.mat',
                mapped_raw_rggb_chs_mat)

      # save visualization raw image
      raw_rggb_chs_vis = (from_rggb_to_rgb(raw_rggb_chs.copy()) * 0.9) ** (
          1 / 1.6)

      imwrite(vis_path + temp_img_name + postfix + '.jpg',
              raw_rggb_chs_vis)

      if mapping_path is not None:
        mapped_raw_rggb_chs_vis = (from_rggb_to_rgb(
          mapped_raw_rggb_chs.copy()) * 0.9) ** (1 / 1.6)

        imwrite(vis_mapping_path + temp_img_name + postfix + '.jpg',
                mapped_raw_rggb_chs_vis)

      print(temp_img_name)
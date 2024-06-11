# Reproduced from https://github.com/deroneriksson/python-wsi-preprocessing/blob/master/docs/wsi-preprocessing-in-python/index.md
from __future__ import division
import glob
from glob import glob as glob_function
import math
import numpy as np
import pandas as pd
import openslide
from openslide import OpenSlideError
import os
import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import re
import util

from util import Time

import skimage.morphology as sk_morphology

import argparse

SCALE_FACTOR = 32 

THUMBNAIL_SIZE = 300
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True

TISSUE_HIGH_THRESH = 20
TISSUE_LOW_THRESH = 0

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 1 
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 1 # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SUMMARY_TITLE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 35
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 20
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 1
TILE_TEXT_H_BORDER = 1

HSV_PURPLE = 270
HSV_PINK = 330
def open_image(filename):
  """
  Open an image (*.jpg, *.png, etc).

  Args:
    filename: Name of the image file.

  returns:
    A PIL.Image.Image object representing an image.
  """
  image = Image.open(filename)
  return image

def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).

  Args:
    filename: Name of the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


def open_image_np(filename):
  """
  Open an image (*.jpg, *.png, etc) as an RGB NumPy array.

  Args:
    filename: Name of the image file.

  returns:
    A NumPy representing an RGB image.
  """
  pil_img = open_image(filename)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def get_training_slide_path(slide_id):
  """
  Convert slide number to a path to the corresponding WSI training slide file.

  Example:
    5 -> ../data/training_slides/TUPAC-TR-005.svs

  Args:
    slide_number: The slide number.

  Returns:
    Path to the WSI training slide file.
  """
  padded_sl_num = slide_id
  slide_filepath = os.path.join(src_train_dir, "" + padded_sl_num + "." + "svs")
  return slide_filepath


def get_tile_image_path(tile, w, h):
  """
  Obtain tile image path based on tile information such as row, column, row pixel position, column pixel position,
  pixel width, and pixel height.

  Args:
    tile: Tile object.

  Returns:
    Path to image tile.
  """
  t = tile
  # padded_sl_num = str(t.slide_num).zfill(3)
  padded_sl_num = t.slide_id
  tile_path = os.path.join(tile_dir, 
                           "" + padded_sl_num + "-" + "tile" + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                            t.r, t.c, t.o_c_s, t.o_r_s, w, h) + "." + "png")
  return tile_path


def get_tile_image_path_by_slide_row_col(slide_number, row, col):
  """
  Obtain tile image path using wildcard lookup with slide number, row, and column.

  Args:
    slide_number: The slide number.
    row: The row.
    col: The column.

  Returns:
    Path to image tile.
  """
  padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = ""
  wilcard_path = os.path.join(tile_dir, padded_sl_num,
                              "" + padded_sl_num + "-" + "tile" + "-r%d-c%d-*." % (
                                row, col) + "png")
  img_path = glob.glob(wilcard_path)[0]
  return img_path


def get_training_image_path(slide_id, large_w=None, large_h=None, small_w=None, small_h=None):
  """
  Convert slide number and optional dimensions to a training image path. If no dimensions are supplied,
  the corresponding file based on the slide number will be looked up in the file system using a wildcard.

  Example:
    5 -> ../data/training_png/TUPAC-TR-005-32x-49920x108288-1560x3384.png

  Args:
    slide_number: The slide number.
    large_w: Large image width.
    large_h: Large image height.
    small_w: Small image width.
    small_h: Small image height.

  Returns:
     Path to the image file.
  """
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id
  if large_w is None and large_h is None and small_w is None and small_h is None:
    wildcard_path = os.path.join(dest_train_dir, "" + padded_sl_num + "*." + "png")
    img_path = glob.glob(wildcard_path)[0]
  else:
    img_path = os.path.join(dest_train_dir, "" + padded_sl_num + "-" + str(
      SCALE_FACTOR) + "x-" + "" + str(
      large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + "png")
  return img_path

def get_filter_image_path(slide_number, slide_id, filter_number, filter_name_info):
  """
  Convert slide number, filter number, and text to a path to a filter image file.

  Example:
    5, 1, "rgb" -> ../data/filter_png/TUPAC-TR-005-001-rgb.png

  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.

  Returns:
    Path to the filter image file.
  """
  dir = filter_dir
  if not os.path.exists(dir):
    os.makedirs(dir)
  img_path = os.path.join(dir, get_filter_image_filename(slide_number, slide_id, filter_number, filter_name_info))
  return img_path

def get_filter_image_filename(slide_number, slide_id, filter_number, filter_name_info, thumbnail=False):
  """
  Convert slide number, filter number, and text to a filter file name.

  Example:
    5, 1, "rgb", False -> TUPAC-TR-005-001-rgb.png
    5, 1, "rgb", True -> TUPAC-TR-005-001-rgb.jpg

  Args:
    slide_number: The slide number.
    filter_number: The filter number.
    filter_name_info: Descriptive text describing filter.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The filter image or thumbnail file name.
  """
  
  ext = "png"
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id
  padded_fi_num = str(filter_number).zfill(3)
  img_filename = "" + padded_sl_num + "-" + padded_fi_num + "-" + "" + filter_name_info + "." + ext
  return img_filename


def get_tile_summary_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary image file.

  Example:
    5 -> ../data/tile_summary_png/TUPAC-TR-005-tile_summary.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary image file.
  """
  # if not os.path.exists(tile_summary_dir):
  #   os.makedirs(tile_summary_dir)
  img_path = os.path.join(tile_summary_dir, get_tile_summary_image_filename(slide_number))
  return img_path

def get_tile_summary_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a tile summary on original image file.

  Example:
    5 -> ../data/tile_summary_on_original_png/TUPAC-TR-005-tile_summary.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile summary on original image file.
  """
  # if not os.path.exists(tile_summary_on_original_dir):
  #   os.makedirs(tile_summary_on_original_dir)
  img_path = os.path.join(tile_summary_on_original_dir, get_tile_summary_image_filename(slide_number))
  return img_path

def get_top_tiles_on_original_image_path(slide_number):
  """
  Convert slide number to a path to a top tiles on original image file.

  Example:
    5 -> ../data/top_tiles_on_original_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles on original image file.
  """
  if not os.path.exists(top_tiles_on_original_dir):
    os.makedirs(top_tiles_on_original_dir)
  img_path = os.path.join(top_tiles_on_original_dir, get_top_tiles_image_filename(slide_number))
  return img_path


def get_tile_summary_image_filename(slide_id, thumbnail=False):
  """
  Convert slide number to a tile summary image file name.

  Example:
    5, False -> TUPAC-TR-005-tile_summary.png
    5, True -> TUPAC-TR-005-tile_summary.jpg

  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The tile summary image file name.
  """

  ext = "png"
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id
  training_img_path = get_training_image_path(slide_id)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = "" + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + "tile_summary" + "." + ext

  return img_filename


def get_top_tiles_image_filename(slide_id, thumbnail=False):
  """
  Convert slide number to a top tiles image file name.

  Example:
    5, False -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png
    5, True -> TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.jpg

  Args:
    slide_number: The slide number.
    thumbnail: If True, produce thumbnail filename.

  Returns:
    The top tiles image file name.
  """
  
  ext = "png"
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id

  training_img_path = get_training_image_path(slide_id)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_filename = "" + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + top_tiles_suffix + "." + ext

  return img_filename

def get_top_tiles_image_path(slide_number):
  """
  Convert slide number to a path to a top tiles image file.

  Example:
    5 -> ../data/top_tiles_png/TUPAC-TR-005-32x-49920x108288-1560x3384-top_tiles.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the top tiles image file.
  """
  if not os.path.exists(top_tiles_dir):
    os.makedirs(top_tiles_dir)
  img_path = os.path.join(top_tiles_dir, get_top_tiles_image_filename(slide_number))
  return img_path

def get_tile_data_filename(slide_id):
  """
  Convert slide number to a tile data file name.

  Example:
    5 -> TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

  Args:
    slide_number: The slide number.

  Returns:
    The tile data file name.
  """
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id
  training_img_path = get_training_image_path(slide_id)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  data_filename = "" + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
    large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + "tile_data" + ".csv"

  return data_filename


def get_tile_data_path(slide_id):
  """
  Convert slide number to a path to a tile data file.

  Example:
    5 -> ../data/tile_data/TUPAC-TR-005-32x-49920x108288-1560x3384-tile_data.csv

  Args:
    slide_number: The slide number.

  Returns:
    Path to the tile data file.
  """
  if not os.path.exists(tile_data_dir):
    os.makedirs(tile_data_dir)
  file_path = os.path.join(tile_data_dir, get_tile_data_filename(slide_id))
  return file_path


def get_filter_image_result(slide_id):
  """
  Convert slide number to the path to the file that is the final result of filtering.

  Example:
    5 -> ../data/filter_png/TUPAC-TR-005-32x-49920x108288-1560x3384-filtered.png

  Args:
    slide_number: The slide number.

  Returns:
    Path to the filter image file.
  """
  # padded_sl_num = str(slide_number).zfill(3)
  padded_sl_num = slide_id
  training_img_path = get_training_image_path(slide_id)
  large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
  img_path = os.path.join(filter_dir, "" + padded_sl_num + "-" + str(
    SCALE_FACTOR) + "x-" + "" + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
    small_h) + "-" + "filtered" + "." + "png")
  return img_path


def parse_dimensions_from_image_filename(filename):
  """
  Parse an image filename to extract the original width and height and the converted width and height.

  Example:
    "TUPAC-TR-011-32x-97103x79079-3034x2471-tile_summary.png" -> (97103, 79079, 3034, 2471)

  Args:
    filename: The image filename.

  Returns:
    Tuple consisting of the original width, original height, the converted width, and the converted height.
  """
  m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
  large_w = int(m.group(1))
  large_h = int(m.group(2))
  small_w = int(m.group(3))
  small_h = int(m.group(4))
  return large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel, large_dimensions):
  """
  Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.

  Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.

  Returns:
    Tuple consisting of the scaled-up width and height.
  """
  small_x, small_y = small_pixel
  large_w, large_h = large_dimensions
  large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
  large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
  return round(large_x), round(large_y)


def training_slide_to_image(slide_id):
  """
  Convert a WSI training slide to a saved scaled-down image in a format such as jpg or png.

  Args:
    slide_number: The slide number.
  """

  img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_id)

  img_path = get_training_image_path(slide_id, large_w, large_h, new_w, new_h)
  print("Saving image to: " + img_path)
  if not os.path.exists(dest_train_dir):
    os.makedirs(dest_train_dir)
  img.save(img_path)

def slide_to_scaled_pil_image(slide_id):
  """
  Convert a WSI training slide to a scaled-down PIL image.

  Args:
    slide_number: The slide number.

  Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
  """
  slide_filepath = get_training_slide_path(slide_id)
  print("Opening Slide #%s: %s" % (slide_id, slide_filepath))
  slide = open_slide(slide_filepath)

  large_w, large_h = slide.dimensions
  new_w = math.floor(large_w / SCALE_FACTOR)
  new_h = math.floor(large_h / SCALE_FACTOR)
  level = slide.get_best_level_for_downsample(SCALE_FACTOR)
  # print(level)
  whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
  whole_slide_image = whole_slide_image.convert("RGB")
  img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
  return img, large_w, large_h, new_w, new_h


def slide_to_scaled_np_image(slide_number):
  """
  Convert a WSI training slide to a scaled-down NumPy image.

  Args:
    slide_number: The slide number.

  Returns:
    Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
  """
  pil_img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_number)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img, large_w, large_h, new_w, new_h

def get_num_training_slides():
  """
  Obtain the total number of WSI training slide images.

  Returns:
    The total number of WSI training slide images.
  """
  num_training_slides = len(glob.glob1(src_train_dir, "*." + "svs"))
  slide_ids = glob_function(os.path.join(src_train_dir, '*.svs'))
  slide_ids_cleaned = [(lambda x: os.path.basename(x).replace(".svs", ""))(x) for x in slide_ids]
  print(slide_ids_cleaned)
  return num_training_slides, slide_ids_cleaned


def training_slide_range_to_images(start_ind, end_ind, slide_ids):
  """
  Convert a range of WSI training slides to smaller images (in a format such as jpg or png).

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).

  Returns:
    The starting index and the ending index of the slides that were converted.
  """
  for slide_num in range(start_ind, end_ind):
    training_slide_to_image(slide_ids[slide_num])
  return (start_ind, end_ind)


def singleprocess_training_slides_to_images():
  """
  Convert all WSI training slides to smaller images using a single process.
  """
  t = Time()

  num_train_images, slide_ids = get_num_training_slides()
  slide_ids = sorted(slide_ids)
  training_slide_range_to_images(0, num_train_images, slide_ids)

  t.elapsed_display()

  """
  FILTERING STARTS HERE

  """

def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8).
  """
  t = Time()

  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def filter_threshold(np_img, threshold, output_type="bool"):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.

  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Threshold", t.elapsed())
  return result


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
  """
  Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
  and eosin are purplish and pinkish, which do not have much green to them.

  Args:
    np_img: RGB image as a NumPy array.
    green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
  """
  t = Time()

  g = np_img[:, :, 1]
  gr_ch_mask = (g < green_thresh) & (g > 0)
  mask_percentage = mask_percent(gr_ch_mask)
  if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
    new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
    print(
      "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
        mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
    gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
  np_img = gr_ch_mask

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  util.np_info(np_img, "Filter Green Channel", t.elapsed())
  return np_img


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
  """
  Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
  red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_lower_thresh: Red channel lower threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_upper_thresh: Blue channel upper threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Red", t.elapsed())
  return result


def filter_red_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out red pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
           filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
           filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
           filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
           filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
           filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
           filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
           filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
           filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Red Pen", t.elapsed())
  return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
  """
  Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
  red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
  Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
  lower threshold value rather than a blue channel upper threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_lower_thresh: Green channel lower threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Green", t.elapsed())
  return result


def filter_green_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out green pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Green Pen", t.elapsed())
  return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
  """
  Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
  red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

  Args:
    rgb: RGB image as a NumPy array.
    red_upper_thresh: Red channel upper threshold value.
    green_upper_thresh: Green channel upper threshold value.
    blue_lower_thresh: Blue channel lower threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_np_info: If True, display NumPy array info and filter time.

  Returns:
    NumPy array representing the mask.
  """
  if display_np_info:
    t = Time()
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_np_info:
    util.np_info(result, "Filter Blue", t.elapsed())
  return result


def filter_blue_pen(rgb, output_type="bool"):
  """
  Create a mask to filter out blue pen marks from a slide.

  Args:
    rgb: RGB image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing the mask.
  """
  t = Time()
  result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Blue Pen", t.elapsed())
  return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
  """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.

  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
  t = Time()
  (h, w, c) = rgb.shape

  rgb = rgb.astype(int)
  rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
  rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
  gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
  result = ~(rg_diff & rb_diff & gb_diff)

  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  util.np_info(result, "Filter Grays", t.elapsed())
  return result


def uint8_to_bool(np_img):
  """
  Convert NumPy array of uint8 (255,0) values to bool (True,False) values

  Args:
    np_img: Binary image as NumPy array of uint8 (255,0) values.

  Returns:
    NumPy array of bool (True,False) values.
  """
  result = (np_img / 255).astype(bool)
  return result


def apply_image_filters(np_img, slide_num=None, slide_id = None, info=None, save=False, display=False):
  """
  Apply filters to image as NumPy array and optionally save and/or display filtered images.

  Args:
    np_img: Image as NumPy array.
    slide_num: The slide number (used for saving/displaying).
    info: Dictionary of slide information (used for HTML display).
    save: If True, save image.
    display: If True, display image.

  Returns:
    Resulting filtered image as a NumPy array.
  """
  rgb = np_img
  save_display(save, display, info, rgb, slide_num, slide_id, 1, "Original", "rgb")

  mask_not_green = filter_green_channel(rgb)
  rgb_not_green = util.mask_rgb(rgb, mask_not_green)
  save_display(save, display, info, rgb_not_green, slide_num, slide_id, 2, "Not Green", "rgb-not-green")

  mask_not_gray = filter_grays(rgb)
  rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)
  save_display(save, display, info, rgb_not_gray, slide_num, slide_id, 3, "Not Gray", "rgb-not-gray")

  mask_no_red_pen = filter_red_pen(rgb)
  rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)
  save_display(save, display, info, rgb_no_red_pen, slide_num, slide_id, 4, "No Red Pen", "rgb-no-red-pen")

  mask_no_green_pen = filter_green_pen(rgb)
  rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)
  save_display(save, display, info, rgb_no_green_pen, slide_num, slide_id, 5, "No Green Pen", "rgb-no-green-pen")

  mask_no_blue_pen = filter_blue_pen(rgb)
  rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)
  save_display(save, display, info, rgb_no_blue_pen, slide_num, slide_id, 6, "No Blue Pen", "rgb-no-blue-pen")

  mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
  rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)
  save_display(save, display, info, rgb_gray_green_pens, slide_num, slide_id, 7, "Not Gray, Not Green, No Pens",
               "rgb-no-gray-no-green-no-pens")

  mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")
  rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)
  save_display(save, display, info, rgb_remove_small, slide_num, slide_id, 8,
               "Not Gray, Not Green, No Pens,\nRemove Small Objects",
               "rgb-not-green-not-gray-no-pens-remove-small")

  img = rgb_remove_small
  print('DONE FILTERING')
  return img


def apply_filters_to_image(slide_num, slide_id, save=True, display=False):
  """
  Apply a set of filters to an image and optionally save and/or display filtered images.

  Args:
    slide_num: The slide number.
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
    (used for HTML page generation).
  """
  t = Time()
  print("Processing slide #%s" % slide_id)

  info = dict()

  if save and not os.path.exists(filter_dir):
    os.makedirs(filter_dir)
  img_path = get_training_image_path(slide_id)
  np_orig = open_image_np(img_path)
  filtered_np_img = apply_image_filters(np_orig, slide_num, slide_id, info, save=save, display=display)

  if save:
    t1 = Time()
    result_path = get_filter_image_result(slide_id)
    pil_img = util.np_to_pil(filtered_np_img)
    pil_img.save(result_path)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))

    t1 = Time()
  print("Slide #%03s processing time: %s\n" % (slide_id, str(t.elapsed())))

  return filtered_np_img, info


def save_display(save, display, info, np_img, slide_num, slide_id, filter_num, display_text, file_text,
                 display_mask_percentage=True):
  """
  Optionally save an image and/or display the image.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    info: Dictionary to store filter information.
    np_img: Image as a NumPy array.
    slide_num: The slide number.
    filter_num: The filter number.
    display_text: Filter display name.
    file_text: Filter name for file.
    display_mask_percentage: If True, display mask percentage on displayed slide.
  """
  mask_percentage = None
  if display_mask_percentage:
    mask_percentage = mask_percent(np_img)
    display_text = display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
  if slide_num is None and filter_num is None:
    pass
  elif filter_num is None:
    display_text = "S%03d " % slide_num + display_text
  elif slide_num is None:
    display_text = "F%03d " % filter_num + display_text
  else:
    display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
  # if display:
  #   util.display_img(np_img, display_text)
  if save:
    save_filtered_image(np_img, slide_num, slide_id, filter_num, file_text)
  if info is not None:
    info[slide_num * 1000 + filter_num] = (slide_num, filter_num, display_text, file_text, mask_percentage)


def mask_percentage_text(mask_percentage):
  """
  Generate a formatted string representing the percentage that an image is masked.

  Args:
    mask_percentage: The mask percentage.

  Returns:
    The mask percentage formatted as a string.
  """
  return "%3.2f%%" % mask_percentage


def save_filtered_image(np_img, slide_num, slide_id, filter_num, filter_text):
  """
  Save a filtered image to the file system.

  Args:
    np_img: Image as a NumPy array.
    slide_num:  The slide number.
    filter_num: The filter number.
    filter_text: Descriptive text to add to the image filename.
  """
  t = Time()
  filepath = get_filter_image_path(slide_num, slide_id, filter_num, filter_text)
  pil_img = util.np_to_pil(np_img)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

  t1 = Time()

def apply_filters_to_image_range(start_ind, end_ind, slide_id, save, display):
  """
  Apply filters to a range of images.

  Args:
    start_ind: Starting index (inclusive).
    end_ind: Ending index (inclusive).
    save: If True, save filtered images.
    display: If True, display filtered images to screen.

  Returns:
    Tuple consisting of 1) staring index of slides converted to images, 2) ending index of slides converted to images,
    and 3) a dictionary of image filter information.
  """
  html_page_info = dict()
  for slide_num in range(start_ind, end_ind):
    _, info = apply_filters_to_image(slide_num, slide_id[slide_num], save=save, display=display)
    html_page_info.update(info)
  return start_ind, end_ind, html_page_info


def singleprocess_apply_filters_to_images(save=True, display=False, html=False, image_num_list=None):
  """
  Apply a set of filters to training images and optionally save and/or display the filtered images.

  Args:
    save: If True, save filtered images.
    display: If True, display filtered images to screen.
    html: If True, generate HTML page to display filtered images.
    image_num_list: Optionally specify a list of image slide numbers.
  """
  t = Time()
  print("Applying filters to images\n")
  
  num_training_slides, slide_ids = get_num_training_slides()
  (s, e, info) = apply_filters_to_image_range(0, num_training_slides, slide_ids, save, display)

  print("Time to apply filters to all images: %s\n" % str(t.elapsed()))

"""
TILING STARTS HERE
"""


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
  a column tile size.

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
  """
  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
  """
  Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).

  Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.

  Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
  """
  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
  """
  Create a PIL summary image including top title area and right side and bottom padding.

  Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.

  Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
  """
  r = row_tile_size * num_row_tiles + title_area_height
  c = col_tile_size * num_col_tiles
  summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
  # add gray edges so that tile text does not get cut off
  # summary_img.fill(120)
  # color title area white
  summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
  #summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
  summary_img = np_img
  summary = util.np_to_pil(summary_img)
  return summary

def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=False,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):
  """
  Generate summary images/thumbnails showing the top tiles ranked by score.

  Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display top tiles to screen.
    save_summary: If True, save top tiles images.
    show_top_stats: If True, append top tile score stats to image.
    label_all_tiles: If True, label all tiles. If False, label only top tiles.
  """
  z = 0  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = get_training_image_path(slide_num)
  np_orig = open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.all_tiles()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  # summary_title = "Slide %03s Top Tile Summary:" % slide_num
  summary_title = ""
  # summary_txt = summary_title + "\n" + summary_stats(tile_sum)
  summary_txt = ""
  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if display:
    summary.show()
    summary_orig.show()
  # if save_summary:
    # save_top_tiles_image(summary, slide_num)
  save_top_tiles_on_original_image(summary_orig, slide_num)

def tile_border_color(tissue_percentage):
  """
  Obtain the corresponding tile border color for a particular tile tissue percentage.

  Args:
    tissue_percentage: The tile tissue percentage

  Returns:
    The tile border color corresponding to the tile tissue percentage.
  """
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = HIGH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = LOW_COLOR
  else:
    border_color = NONE_COLOR
  return border_color


def summary_title(tile_summary):
  """
  Obtain tile summary title.

  Args:
    tile_summary: TileSummary object.

  Returns:
     The tile summary title.
  """
  return "Slide %03s Tile Summary:" % tile_summary.slide_num

def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
  """
  Draw a border around a tile with width TILE_BORDER_SIZE.

  Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
  """
  for x in range(0, border_size):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

def save_top_tiles_image(pil_img, slide_num):
  """
  Save a top tiles image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_top_tiles_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Image", str(t.elapsed()), filepath))

def save_tile_summary_on_original_image(pil_img, slide_num):
  """
  Save a tile summary on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

  t = Time()

def save_top_tiles_on_original_image(pil_img, slide_num):
  """
  Save a top tiles on original image and thumbnail to the file system.

  Args:
    pil_img: Image as a PIL Image.
    slide_num: The slide number.
  """
  t = Time()
  filepath = get_top_tiles_on_original_image_path(slide_num)
  pil_img.save(filepath)
  print("%-20s | Time: %-14s  Name: %s" % ("Save Top Orig", str(t.elapsed()), filepath))

  t = Time()

def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  """
  Obtain a PIL image representation of text.

  Args:
    text: The text to convert to an image.
    w_border: Tile text width border (left and right).
    h_border: Tile text height border (top and bottom).
    font_path: Path to font.
    font_size: Size of font.
    text_color: Tile text color.
    background: Tile background color.

  Returns:
    PIL image representing the text.
  """

  font = ImageFont.truetype(font_path, font_size)
  x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  draw.text((w_border, h_border), text, text_color, font=font)
  return image

class TileSummary:
  """
  Class for tile summary information.
  """

  slide_num = None
  orig_w = None
  orig_h = None
  orig_tile_w = None
  orig_tile_h = None
  scaled_w = None
  scaled_h = None
  scaled_tile_w = None
  scaled_tile_h = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w = scaled_tile_w
    self.scaled_tile_h = scaled_tile_h
    self.tissue_percentage = tissue_percentage
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tiles = []
  def mask_percentage(self):
    """
    Obtain the percentage of the slide that is masked.

    Returns:
       The amount of the slide that is masked as a percentage.
    """
    return 100 - self.tissue_percentage

  def num_tiles(self):
    """
    Retrieve the total number of tiles.

    Returns:
      The total number of tiles (number of rows * number of columns).
    """
    return self.num_row_tiles * self.num_col_tiles

  def tiles_by_tissue_percentage(self):
    """
    Retrieve the tiles ranked by tissue percentage.

    Returns:
       List of the tiles ranked by tissue percentage.
    """
    sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_list

  def tiles_by_score(self):
    """
    Retrieve the tiles ranked by score.

    Returns:
       List of the tiles ranked by score.
    """
    # sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    sorted_list = self.tiles
    return sorted_list

  def all_tiles(self):
    """
    Retrieve the all tiles that have at least x tissue percent.

    Returns:
       List of the all tiles that have at least x tissue percent.
    """
    sorted_tiles = self.tiles_by_score()
    all_tiles = sorted_tiles
    return all_tiles

  def get_tile(self, row, col):
    """
    Retrieve tile by row and column.

    Args:
      row: The row
      col: The column

    Returns:
      Corresponding Tile object.
    """
    tile_index = (row - 1) * self.num_col_tiles + (col - 1)
    tile = self.tiles[tile_index]
    return tile


class Tile:
  """
  Class for information about a tile.
  """

  def __init__(self, tile_summary, slide_num, slide_id, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p):
    self.tile_summary = tile_summary
    self.slide_num = slide_num
    self.slide_id = slide_id

    self.np_scaled_tile = np_scaled_tile
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.r_s = r_s
    self.r_e = r_e
    self.c_s = c_s
    self.c_e = c_e
    self.o_r_s = o_r_s
    self.o_r_e = o_r_e
    self.o_c_s = o_c_s
    self.o_c_e = o_c_e
    self.tissue_percentage = t_p

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def save_tiles(self):
    t = self
    slide_filepath = get_training_slide_path(t.slide_id)
    s = open_slide(slide_filepath)
  
    x, y = t.o_c_s, t.o_r_s
    w, h = ROW_TILE_SIZE, COL_TILE_SIZE
    tile_region = s.read_region((x, y), WSI_LEVEL, (w, h))
    pil_img = tile_region.convert("RGB")
    img2_resized = np.asarray(pil_img.resize((224, 224)))
    img2_resized = Image.fromarray(img2_resized, 'RGB')
    img_path = get_tile_image_path(self, w, h)
    img_path = img_path.replace('w' + str(w) + '-h' + str(h))
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    img2_resized.save(img_path)

def singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=False,
                                           html=True, image_num_list=None):
  
  num_training_slides, slide_ids = get_num_training_slides()
  for slide_num in range(num_training_slides):
    img_path = get_filter_image_result(slide_ids[slide_num])
    np_img = open_image_np(img_path)

    tile_sum = score_tiles(slide_num, slide_ids[slide_num], np_img)
    if save_summary:
      generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
    if save_top_tiles:
      for tile in tile_sum.all_tiles():
        tile.save_tiles()

def score_tiles(slide_num, slide_id, np_img=None, dimensions=None, small_tile_in_tile=False):
  """
  Score all tiles for a slide and return the results in a TileSummary object.

  Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.

  Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
  """
  if dimensions is None:
    img_path = get_filter_image_result(slide_id)
    o_w, o_h, w, h = parse_dimensions_from_image_filename(img_path)
  else:
    o_w, o_h, w, h = dimensions

  if np_img is None:
    np_img = open_image_np(img_path)

  slide_filepath = get_training_slide_path(slide_id)
  s = open_slide(slide_filepath)
  mpp = float(s.properties['aperio.MPP'])
  if mpp >= 0.5:
    ROW_TILE_SIZE, COL_TILE_SIZE = 1024, 1024
    MAG_FACTOR = 1
  elif mpp < 0.5:
    ROW_TILE_SIZE, COL_TILE_SIZE = 512, 512
    MAG_FACTOR = 4

  row_tile_size = round(ROW_TILE_SIZE*MAG_FACTOR / SCALE_FACTOR)  
  col_tile_size = round(COL_TILE_SIZE*MAG_FACTOR / SCALE_FACTOR) 

  num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

  tile_sum = TileSummary(slide_num=slide_id,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=tissue_percent(np_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
  for t in tile_indices:
    count += 1  # tile_num
    r_s, r_e, c_s, c_e, r, c = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    t_p = tissue_percent(np_tile)

    o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
    o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))

    # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
    if (o_c_e - o_c_s) > COL_TILE_SIZE:
      o_c_e -= 1
    if (o_r_e - o_r_s) > ROW_TILE_SIZE:
      o_r_e -= 1

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, slide_num, slide_id, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                o_c_e, t_p)

    if t_p >= TISSUE_HIGH_THRESH:
      tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--wsi_level', type = int, default = 1, help = 'whole slide level to extract the tiles from')
  parser.add_argument('--tile_size', type = int, default = 512, help = 'tile size')
  parser.add_argument('--path_to_wsi_images', type = str, default = None, help = 'Parent path to all the WSIs')
  parser.add_argument('--path_to_generated_tiles', type = str, default = None, help = 'Parent path to the generated tiles')
  args = parser.parse_args()

  ROW_TILE_SIZE, COL_TILE_SIZE = args.tile_size, args.tile_size
  WSI_LEVEL = args.wsi_level
  wsi_root_path = args.path_to_wsi_images
  wsi_tiles_root_dir = args.path_to_generated_tiles

  wsi_cases_dir = glob_function(os.path.join(wsi_root_path, '*'))
  wsi_cases_dir = sorted(wsi_cases_dir)

  for case in range(len(wsi_cases_dir)):
    wsi_tiles_cases_dir = os.path.join(wsi_tiles_root_dir, os.path.basename(wsi_cases_dir[case]))
    
    if not os.path.exists(wsi_tiles_cases_dir):
      os.makedirs(wsi_tiles_cases_dir)
      
    base_dir = wsi_tiles_cases_dir
    src_train_dir = wsi_cases_dir[case]
    dest_train_dir = os.path.join(base_dir, "low_resolution_" + "png")
    filter_dir = os.path.join(base_dir, "filter_" + "png")
    tile_summary_dir = os.path.join(base_dir, "tile_summary_" + "png")
    tile_summary_on_original_dir = os.path.join(base_dir, "tile_summary_on_original_" + "png")
    tile_data_dir = os.path.join(base_dir, "tile_data")

    top_tiles_suffix = "top_tile_summary"
    top_tiles_dir = os.path.join(base_dir, top_tiles_suffix + "_" + "png")
    top_tiles_on_original_dir = os.path.join(base_dir, top_tiles_suffix + "_on_original_" + "png")

    tile_dir = os.path.join(base_dir, "tiles_png")
    singleprocess_training_slides_to_images()
    singleprocess_apply_filters_to_images(html = False)
    singleprocess_filtered_images_to_tiles(display = False, image_num_list = None, save_summary=False, save_data=False, save_top_tiles=True)

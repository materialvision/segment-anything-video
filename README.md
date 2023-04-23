<div align="center">
<h2>
     MetaSeg: Packaged version of the Segment Anything repository
</h2>
<div>
    <img width="1000" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.2.2/metaseg_demo.gif">
</div>
    <a href="https://pepy.tech/project/metaseg"><img src="https://pepy.tech/badge/metaseg" alt="downloads"></a>
    <a href="https://badge.fury.io/py/metaseg"><img src="https://badge.fury.io/py/metaseg.svg" alt="pypi version"></a>
    <a href="https://huggingface.co/spaces/ArtGAN/metaseg-webui"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="HuggingFace Spaces"></a>

</div>

This repo is a packaged version of the [segment-anything](https://github.com/facebookresearch/segment-anything) model.

### Installation
```bash
pip install metaseg
```

### Usage
```python
from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

# CLI Masker

The `cli_masker.py` script is a command-line interface for segmenting objects in images and videos using the SegAutoMaskPredictor class. It allows you to generate object masks for images and videos and save the results as individual files.

## Prerequisites

- Python 3.6 or higher
- OpenCV
- PyTorch
- torchvision
- tqdm
- Pillow

## Usage

To use the `cli_masker.py` script, you need to provide the required arguments as follows:

For images:

```bash
python cli_masker.py /path/to/image.jpg --model_type vit_l --points_per_side 16 --points_per_batch 64 --min_area 1000 --output_folder /path/to/output/folder --show --save

For videos:

python cli_masker.py /path/to/video.mov --model_type vit_l --points_per_side 16 --points_per_batch 64 --min_area 1000 --output_folder /path/to/output/folder

Arguments
source: The path to the input image or video file.
--model_type: The type of model to use for segmentation. Options are vit_l, vit_h, and vit_b. (default: vit_l)
--points_per_side: The number of points per side to sample for the mask generation process. (default: 16)
--points_per_batch: The number of points to process in each batch during mask generation. (default: 64)
--min_area: The minimum area for a mask to be considered valid. (default: 1000)
--output_folder: The path to the folder where the output files will be saved. (default: output)
--show: (Image only) Display the result. (optional)
--save: (Image only) Save the result. (optional)
Output
The script will output individual files for each segmented object. In the case of images, the output will be image files with segmented object masks. For videos, the output will be video files with object masks for each object.

The output files will be saved in the specified output_folder and will be named as object_{i}.ext where {i} is the object index and ext is the file extension (.jpg for images and .mp4 for videos).

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.

# For image

results = SegAutoMaskPredictor().image_predict(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=0,
    output_path="output.jpg",
    show=True,
    save=False,
)

# For video

results = SegAutoMaskPredictor().video_predict(
    source="video.mp4",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=1000,
    output_path="output.mp4",
)

# For manuel box and point selection

# For image
results = SegManualMaskPredictor().image_predict(
    source="image.jpg",
    model_type="vit_l", # vit_l, vit_h, vit_b
    input_point=[[100, 100], [200, 200]],
    input_label=[0, 1],
    input_box=[100, 100, 200, 200], # or [[100, 100, 200, 200], [100, 100, 200, 200]]
    multimask_output=False,
    random_color=False,
    show=True,
    save=False,
)

# For video

results = SegManualMaskPredictor().video_predict(
    source="test.mp4",
    model_type="vit_l", # vit_l, vit_h, vit_b
    input_point=[0, 0, 100, 100]
    input_label=N
    input_box=None,
    multimask_output=False,
    random_color=False,
    output_path="output.mp4",
)
```

### SAHI + Segment Anything

```python
from metaseg import sahi_sliced_predict, SahiAutoSegmentation

image_path = "test.jpg"
boxes = sahi_sliced_predict(
    image_path=image_path,
    detection_model_type="yolov5", #yolov8, detectron2, mmdetection, torchvision
    detection_model_path="yolov5l6.pt",
    conf_th=0.25,
    image_size=1280,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

SahiAutoSegmentation().predict(
    source=image_path,
    model_type="vit_b",
    input_box=boxes,
    multimask_output=False,
    random_color=False,
    show=True,
    save=False,
)
```

<img width="700" alt="teaser" src="https://github.com/kadirnar/segment-anything-pip/releases/download/v0.5.0/sahi_autoseg.png">

# Extra Features

- [x] Support for Yolov5/8, Detectron2, Mmdetection, Torchvision models
- [x] Support for video and web application(Huggingface Spaces)
- [x] Support for manual single multi box and point selection
- [x] Support for pip installation
- [x] Support for SAHI library

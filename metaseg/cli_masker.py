from typing import Any, Dict, List, Optional

import os
import argparse
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

from metaseg import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from metaseg.utils import (
    download_model,
    load_box,
    load_image,
    load_mask,
    load_video,
    multi_boxes,
    save_image,
    show_image,
)

def custom_load_video(source: str):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file {source}")

    return cap

class SegAutoMaskPredictor:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_type):
        if self.model is None:
            self.model_path = download_model(model_type)
            self.model = sam_model_registry[model_type](checkpoint=self.model_path)
            self.model.to(device=self.device)

        return self.model

    def image_predict(
        self,
        source,
        model_type,
        points_per_side,
        points_per_batch,
        min_area,
        output_folder="output",
        show=False,
        save=False,
    ):
        read_image = load_image(source)
        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
        )

        masks = mask_generator.generate(read_image)

        sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        mask_image = np.zeros((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3), dtype=np.uint8)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i, ann in enumerate(sorted_anns):
            m = ann["segmentation"]
            img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            img[m > 0] = [255, 255, 255]  # Set the mask color to white
            output_path = os.path.join(output_folder, f"object_{i + 1}.png")
            save_image(output_path=output_path, output_image=img)

        return masks
    
    def video_predict(self, source, model_type, points_per_side, points_per_batch, min_area, start_frame, end_frame, output_folder="output"):
        cap = custom_load_video(source)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writers = []  # To store VideoWriter objects
        written_frames = []

        model = self.load_model(model_type)
        mask_generator = SamAutomaticMaskGenerator(
            model, points_per_side=points_per_side, points_per_batch=points_per_batch, min_mask_region_area=min_area
        )

        if end_frame is None:
            end_frame = length

        for frame_idx in tqdm(range(start_frame, end_frame)):
            ret, frame = cap.read()
            if not ret:
                break

            masks = mask_generator.generate(frame)

            if len(masks) == 0:
                continue

            sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)

            for i, ann in enumerate(sorted_anns):
                m = ann["segmentation"]
                img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
                img[m > 0] = [255, 255, 255]  # Set the mask color to white

                if len(video_writers) <= i:
                    output_path = os.path.join(output_folder, f"object_{i + 1}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    video_writers.append(video_writer)
                    written_frames.append(0)

                while written_frames[i] < frame_idx:
                    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    video_writers[i].write(black_frame)
                    written_frames[i] += 1

                video_writers[i].write(img)
                written_frames[i] += 1

        cap.release()

        # Release VideoWriter objects
        for video_writer in video_writers:
            video_writer.release()

        cv2.destroyAllWindows()

        return output_folder

  
def main():
    parser = argparse.ArgumentParser(description="Segment objects from images and videos using Metaseg.")
    parser.add_argument("source", help="Path to the input image or video.")
    parser.add_argument("--model_type", default="vit_l", choices=["vit_l", "vit_h", "vit_b"], help="Model type (vit_l, vit_h, vit_b).")
    parser.add_argument("--points_per_side", type=int, default=16, help="Number of points per side.")
    parser.add_argument("--points_per_batch", type=int, default=64, help="Number of points per batch.")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum area of the objects to be segmented.")
    parser.add_argument("--output_folder", default="output", help="Path to the output folder.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for processing the video.")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for processing the video.")
    parser.add_argument("--show", action="store_true", help="Display the result.")
    parser.add_argument("--save", action="store_true", help="Save the result.")


    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source file '{args.source}' not found.")
        sys.exit(1)

    predictor = SegAutoMaskPredictor()
    
    if os.path.splitext(args.source)[-1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
        # Image
        results = predictor.image_predict(
            source=args.source,
            model_type=args.model_type,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            min_area=args.min_area,
            output_folder=args.output_folder,
            show=args.show,
            save=args.save,
        )
    else:
        # Video
        results = predictor.video_predict(
            source=args.source,
            model_type=args.model_type,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            min_area=args.min_area,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_folder=args.output_folder,
        )


if __name__ == "__main__":
    main()

#python cli_masker.py /path/to/image.jpg --model_type vit_l --points_per_side 16 --points_per_batch 64 --min_area 0 --output_folder /path/to/output/folder --save
#python cli_masker.py /path/to/video.mov --model_type vit_l --points_per_side 16 --points_per_batch 64 --min_area 1000 --output_folder /path/to/output/folder --start_frame=0 --end_frame=None

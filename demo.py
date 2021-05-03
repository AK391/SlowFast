import torch
# Choose the `x3d_s` model 
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

import gradio as gr




#Video
torch.hub.download_url_to_file('https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4', 'archery.mp4')

model_name = 'x3d_s'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

# Set to GPU or CPU
device = "cpu"
model = model.to(device)

json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# Get transform parameters based on model
transform_params = model_transform_params[model_name]

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second

def x3dpred(video):

  # Select the duration of the clip to load by specifying the start and end duration
  # The start_sec should correspond to where the action occurs in the video
  start_sec = 0
  end_sec = start_sec + clip_duration

  # Initialize an EncodedVideo helper class and load the video
  video = EncodedVideo.from_path(video)

  # Load the desired clip
  video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

  # Apply a transform to normalize the video input
  video_data = transform(video_data)

  # Move the inputs to the desired device
  inputs = video_data["video"]
  inputs = inputs.to(device)


  # Pass the input clip through the model
  preds = model(inputs[None, ...])

  # Get the predicted classes
  post_act = torch.nn.Softmax(dim=1)
  preds = post_act(preds)
  pred_classes = preds.topk(k=5).indices[0]

  # Map the predicted classes to the label names
  pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
  return "%s" % ", ".join(pred_class_names)

inputs = gr.inputs.Video(label="Input Video")
outputs = gr.outputs.Textbox(label="Top 5 predicted labels")

title = "X3D"
description = "demo for X3D networks pretrained on the Kinetics 400 dataset. To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2004.04730'>X3D: Expanding Architectures for Efficient Video Recognition</a> | <a href='https://github.com/facebookresearch/pytorchvideo'>Github Repo</a></p>"

examples = [
    ['archery.mp4']
]

gr.Interface(x3dpred, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch(debug=True)
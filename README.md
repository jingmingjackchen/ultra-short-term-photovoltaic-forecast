# Ultra-Short-Term Photovoltaic Forecast

Experiments to use ground-based sky imaging to forecast photovoltaic power output.

## Overview

The intention of this project is to use images/video captured from a fisheye camera on the ground (rooftop) of the sky to predict the movement of clouds, then combine this prediction of cloud coverage with other weather data to generate an ultra-short-term forecast of photovoltaic (solar) power generation output.

## Plan

1. Extract motion vector or optical flow of cloud movements from camera feed.
2. Use machine learning models to predict cloud movements.
   - The input into the machine learning model is previous frame(s) of video and the motion vector/optical flow of the frame(s).
   - The output of the model is the predicted next frame of video.
3. Use another model to calculate the cloud coverage of the predicted frame.
4. Use the predicted cloud coverage data, combined with other weather data, to generate ultra-short-term forecasts of solar panel power output.

So far, this repository only contains code for part 1 and 2.

## Requirements:

- Written and tested on Python 3.10
- Tested in a conda environment with dependecies listed in [requirements.txt](requirements.txt).

## Instructions

1. Create a conda environment:
   `conda create --name <my-env>`
2. Install the dependecies listed in [requirements.txt](requirements.txt):
   `conda install requirements.txt`
3. Activate the conda environment that you had just created.
4. Replace [output_video.mp4](output_video.mp4) with the video captured from camera placed on the rooftop.
5. Run the first cell in [extract_vid_to_npz.ipynb](extract_vid_to_npz.ipynb)
   - Make sure to change the `resize_to` variable to a smaller resolution in order to decrease memory usage during training.
6. Train and evaluate the model:
   - [model_tensorflow_cnn.ipynb](model_tensorflow_cnn.ipynb): Run every cell.
   - [model_pytorch_cnn.ipynb](model_pytorch_cnn.ipynb): Start running cells below `START FROM HERE`.

## Disclaimer

- This project is mostly a proof-of-concept, and is only partially complete.
- I currently have no plans to continue turning this project into a finalized product.

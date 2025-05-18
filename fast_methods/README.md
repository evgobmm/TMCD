# Fast Object Detection Methods

This folder contains an experiment to benchmark different approaches for detecting anomalous objects in video data on a GPU (e.g. Google Colab with an A100).

The following detection strategies are briefly considered:

1. **Deep Learning Detectors** – modern single-stage models such as YOLOv8 or YOLOX provide real-time object detection (hundreds of frames per second on an A100). They require labelled training data or pre-trained weights.
2. **Background Subtraction** – frame differencing and running averages are extremely fast but only handle static cameras and predictable backgrounds.
3. **Mahalanobis Distance** – modelling the distribution of video clips with covariance estimators (e.g. the robust TMCD algorithm). Outliers can be spotted by thresholding the squared Mahalanobis distance.

The supplied notebook demonstrates how to time the Mahalanobis-based approach and serves as a template for comparing against deep-learning baselines.

## Running in Colab

1. Clone this repo in Colab:
```bash
!git clone https://github.com/evgobmm/TMCD.git
%cd TMCD/fast_methods
```
2. Upload `tensor_dataset_sea.zip` to `/content` (see the dataset link in the main README).
3. Open `mahalanobis_benchmark.ipynb` and run all cells on a GPU runtime.

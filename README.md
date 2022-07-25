# ONNX YOLOv7 Object Detection
 Python scripts performing object detection using the YOLOv7 model in ONNX.

![! ONNX YOLOv7 Object Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/blob/main/doc/img/detected_objects.jpg)
*Original image: https://www.flickr.com/photos/nicolelee/19041780*

# Important
- The input images are directly resized to match the input size of the model. I skipped adding the pad to the input image, it might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.

# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection.git
cd ONNX-YOLOv7-Object-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309). Download the models from **[his repository]**(https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7). For that, you can either run the `download_single_batch.sh` or copy the google drive link inside that script in your browser to manually download the file. Then, extract and copy the downloaded onnx models (for example `yolov7-tiny_480x640.onnx`) to your **[models directory](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/tree/main/models)**, and fix the file name in the python scripts accordingly.

- The License of the models is GPL-3.0 license: [License](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)

# Original YOLOv7 model
The original YOLOv7 model can be found in this repository: [YOLOv7 Repository](https://github.com/WongKinYiu/yolov7)
- For **Darknet style model**, check the [darknet branch](https://github.com/WongKinYiu/yolov7/tree/darknet)
- For **YOLOv5 compatible model**, check the [u5 branch](https://github.com/WongKinYiu/yolov7/tree/u5)

# Examples

 * **Image inference**:
 ```
 python image_object_detection.py
 ```

 * **Webcam inference**:
 ```
 python webcam_object_detection.py
 ```

 * **Video inference**: https://youtu.be/yYo0XQp97vo
 ```
 python video_object_detection.py
 ```
 ![!YOLOv7 detection video](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/blob/main/doc/img/yolov7_video.gif)

  *Original video: https://youtu.be/zPre8MgmcHY*

 * **Comparison with YOLOv5 or YOLOv6**: https://youtu.be/WSFmLMLIbDQ
 ```
 python comparison_with_yolov5_v6.py
 ```
![!YOLOv7 Vs YOLOv5 detection video](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/blob/main/doc/img/yolov7_yolov5_video.gif)
![!YOLOv7 Vs YOLOv6 detection video](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/blob/main/doc/img/yolov7_yolov6_video.gif)
  *Original video: https://youtu.be/zPre8MgmcHY*

- Replace the `yolov5_v6_path` with the actual path to the YOLOv5 or YOLOv6 model.
- **Convert YOLOv5 model to ONNX** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V-F3erKkPun-vNn28BoOc6ENKmfo8kDh?usp=sharing)
- **Convert YOLOv6 model to ONNX** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pke1ffMeI2dXkIAbzp6IHWdQ0u8S6I0n?usp=sharing)

# References:
* YOLOv7 model: https://github.com/WongKinYiu/yolov7
* Paper: https://arxiv.org/abs/2207.02696
* YOLOv6 model: https://github.com/WongKinYiu/yolov7
* YOLOv5 model: https://github.com/ultralytics/yolov5
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow

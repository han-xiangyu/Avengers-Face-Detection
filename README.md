## Avengers Face Recognition

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This project is dedicated to recognizing faces of five Avengers, either individually or in a group image. We have implemented four  methods:
1. VGG feature vector extractor coupled with a linear neural network classifier.
2. A finely-tuned VGG16 model.
3. A finely-tuned ResNet model.
4. A finely-tuned Inception-ResNet-V1 model.

The standout performer is our fourth method, achieving an impressive 99.7% accuracy for individual Avenger recognition and 90% in scenarios involving all five Avengers. 
![Screenshot of the performance1](/images/screenshot1.png)
![Screenshot of the performance2](/images/screenshot2.png)

## Table of Contents

- [Packages Requirements](#packages-requirements)
- [Usage](#usage)
- [License](#license)

## Packages Requirements

| Package           | Description                        |
|-------------------|------------------------------------|
| `torch`           | Deep learning library              |
| `pickle`          | Data serialization utility         |
| `matplotlib`      | Graphical plotting library         |
| `numpy`           | Foundation for numerical computing |
| `opencv-python`   | Library for computer vision tasks  |
| `pillow`          | Image processing toolkit           |
| `facenet_pytorch` | Face recognition library           |
| `Ipython`         | Enhanced interactive Python shell  |
| `deepface`        | Deep learning-based face analysis  |
| `yoloface`        | Advanced face detection model      |

## Usage

1. Prepare your image dataset and execute the [shuffle_dataset_into_train_val_test.py](shuffle_dataset_into_train_val_test.py) script to randomly distribute them into training, validation, and test datasets, following an 80%:10%:10% ratio.

2. To train the VGG feature vector extractor and linear neural network classifier (main_v1), as described in the [Description](#description), execute [train_model.py](main_v1/train_model.py).

3. For training the Fine-tuned VGG16 model (main_v2), as outlined in the [Description](#description), run [train_myVGG.py](main_v2/train_myVGG.py).

4. To train the Fine-tuned ResNet model (main_v3), refer to the [Description](#description), and execute `train_model.py`.

5. For the Fine-tuned Inception-ResNet-V1 model (main_v4), as detailed in the [Description](#description), launch [train_myIncepRes.py](main_v4/train_myIncepRes.py).

6. Execute `video_prediction` in each respective folder to perform video-based predictions.

7. For the testset prediction, please run the code `testset_prediction.py` in each respective folder

## License

This project is distributed under the [MIT License](LICENSE).

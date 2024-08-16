
# Vietnamese Sign Language Recognition (PyTorch)

## Tabel of Content

 - [Introduction](#Introduction)
 - [Feature](#Feature)
 - [Project Structure](#Project-Structure)
 - [Installation](#Installation)
 - [Contact](#Contact)
    
## Introduction
This project focuses on developing a system to recognize and interpret Vietnamese Sign Language (VSL) using machine learning and computer vision techniques. The goal is to bridge the communication gap between the deaf community and non-signers in Vietnam.

![Demo](https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-/blob/master/demo/demo.gif)

## Features
- Real-time sign language recognition: Using a webcam or pre-recorded video.
- Support for multiple signs: Recognition of a wide range of Vietnamese Sign Language gestures.
- User-friendly interface: Easy to use for both developers and non-developers.

## Project Structure
```bash
Vietnamese-Sign-Language-Recognition-(PyTorch)/
├── data/alphabet                           # Data files
├── demo/                   		    # Demo
├── images/                   		    # Images
├── src/                      		    # Source code for the project
│   ├── classification.py                   # Script for classification
│   ├── collect_data.py           	    # Script for collecting new data
│   ├── config.py           		    # Script for configuration settings
│   ├── dataset.py           		    # Script for dataset
│   ├── hand_tracking.py           	    # Script for hand detection
│   ├── model.py           		    # Script for CNNs model
│   ├── utils.py                            # Script for snippets
├── trained_models/                         # Saved models and parameters
├── inferent.py          		    # Script for inference
├── requirements.txt          		    # Python packages required
├── train.py          		            # Script for training
├── README.md                 		    # Project documentation
```
## Installation
1. **Clone the repository**
```bash
 git clone https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-.git
 cd Vietnamese-Sign-Language-Recognition-PyTorch-
```
2. **Install the required libraries**
```bash
pip install -r requirements.txt
```
3. **Download the VSL dataset**
The dataset is available [here](https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-/tree/master/dataset/alphabet). After downloading, place it in the dataset/ directory.

![Dataset](https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-/blob/master/images/dataset.png)

Or you can create your own dataset:
```bash
python src/collect_data.py -d [DATA_PATH] -n [NUM_IMAGE]
```
![Collect A data](https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-/blob/master/images/collect%20A%20data.png)
![Collect E data](https://github.com/ndanh318/Vietnamese-Sign-Language-Recognition-PyTorch-/blob/master/images/collect%20E%20data.png)
4. **Train the model (Optional)**
```bash
python train.py
```
5. **Run the recognition system**
```bash
python inference.py
```
## Contact

For any questions or issues, please contact me at ngoduyanh8888@gmail.com.

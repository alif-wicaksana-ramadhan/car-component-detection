# AI Model for Detecting Changes in Car Components

A real-time deep learning system that detects physical state changes in car components (doors and hood) using computer vision and web interface integration.

## 🎯 Project Overview

This project implements an AI system capable of recognizing physical changes in a 3D car model through real-time web interface analysis. The system detects the open/close status of car doors and hood with immediate response to user interactions.

### Key Capabilities

- **Multi-Task Approach**: The model has four heads as output with specific tasks. Each head has to predicts the state of specific object of an image, which are front left door, front right door, rear left door, rear right door, and the hood. All the model heads has shared backbone weights.
- **Visual Captioning Approach**: There is also Visual Language Model that serves a service to give description based on image input.
- **Visual Grounding Approach**: There is also Visual Language Model that serves a service to give bounding box to an image based on user's instruction.
- **Web Integration**: The model was deployed in a FastAPI based server, and served as a service, seamlessly presented as a browser-based real-time inference system.

## 🏗️ System Description

### Task 1 (Mandatory): Multitask Learning (Real-Time Component's State Detection)

- **Architecture:** The proposed model utilizes Inverted Residual Networks to enhance the feature extraction quality while keep the computation complexity stay low. The Inverted Residual Layers are then fused with Feature Pyramid Networks configuration.
- **Dataset Generation:** The dataset was created by record a screen while open and moving the 3d car model continuously.
- **Multi-component detection**: The model is able to detect the states of car's components including Front Left and Right Doors, Rear Left and Right Doors, and Hood.
- **View-invariant detection** The model is able to perform robustly even if the view angle is dynamic.

### Task 2 (Bonus): Visual Captioning (Car Image Descriptor)

- **Arcchitecture**: The proposed solution utilizes BLIP (Bootstrapping Language-Image Pre-training), proposed by J. Li, in 2022. BLIP was chosen as it is one of available state of the art visual captioning model which gain a lot of attention.
- **Dataset Generation:** The labels used in the training process were generated using OpenAI API in order to keep the variations while keeping the meaning.

### Task 3 (Bonus): Visual Grounding

- **Architecture:** The proposed solution utilizes Grounding DINO, published in 2023 by S. Liu, et. al. Grounding DINO was chosen as it is one of available state of the art visual captioning model which gain a lot of attention.
- **Description:** The system receives an image with an instruction to locate some items, for instance the location of opened doors. The system will return an image containing bounding boxes, marking the position of the requested objects.
- **Interactive web interface:** The system also has been integrated with the FastAPI server, and served as an AI service. So, the frontend web application will able to use it.

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended for real-time inference)
- Modern web browser (Chrome/Firefox)
- 8GB+ RAM
```

### Installation & Preparation

1. **Clone Repository**

   ```bash
   https://github.com/alif-wicaksana-ramadhan/car-component-detection.git
   cd car-component-detection
   ```

2. **Setup Environment (Choose One)**

   for using venv

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   for using conda/miniconda

   ```bash
   conda create -n venv python=3.10
   conda activate venv
   pip install -r requirements.txt
   ```

3. **Access 3D Car Model**

   - Navigate to the provided 3D Car Model Web View
   - Ensure interactive buttons for 4 doors + 1 hood are functional

4. **Dataset Creation**
   - Navigate to 'preparation' folder
   ```bash
   cd preparation/
   ```
   - Run the record*specific_window.py <window_name> <output_file>, while moving the Car Model Web View continuously, to record the Car Model Web View. (\_For ease of use, make sure name the output representing car conditions.*)
   ```bash
   python record_specific_window.py edge 00000.avi # 00000 representing all monitored components are closed
   ```
   - Run the extract_video_to_frames.py, to extract frames from the recorded videos in the previous step
   ```bash
   python extract_video_to_frames.py
   ```
   - Run create_dataset.py, to generate metadata for the generated frames according to the car conditions
   ```bash
   python create_dataset.py
   ```

### Usage Guide

1. **Multitask Training**

   - Navigate to multitask_learning folder

   ```bash
   cd multitask_learning/
   ```

   - Run the training code "train.py"

   ```bash
   python train.py
   ```

2. **Fine Tune VLM for Visual Captioning**

   - Navigate to visual_captioning folder

   ```bash
   cd visual_captioning/
   ```

   - Run the fine tuning process by run code train.py

   ```bash
   python train.py
   ```

3. **Fine Tune VLM for Visual Grounding (_Not yet implemented_)**

4. **Run Model Inference as Service**

   - Ensure in root folder
   - Run the FastAPI server by run main.py
   - Run the screen streamer by run screen_streamer.py

5. **Run the frontend application that integrates the Model Inference Service**
   - Navigate to website folder
   ```bash
   cd website/
   ```
   - Run the frontend in development mode
   ```bash
   npm run dev
   ```

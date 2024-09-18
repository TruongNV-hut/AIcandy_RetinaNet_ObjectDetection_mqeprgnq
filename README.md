# RetinaNet and Object detection

<p align="justify">
<strong>RetinaNet</strong>
is a single-stage object detection network introduced by Facebook AI Research in 2017. It addresses class imbalance in dense object detectors through its key innovation, Focal Loss. RetinaNet also incorporates a Feature Pyramid Network for multi-scale feature extraction and uses shared subnetworks for classification and bounding box regression. This architecture enables RetinaNet to achieve high accuracy and efficiency in real-time object detection tasks.
</p>

## Object detection
<p align="justify">
<strong>Object detection</strong> is a computer vision technique in machine learning that involves identifying and locating objects within an image or video. Unlike image classification, which assigns a label to an entire image, object detection not only classifies objects but also draws bounding boxes around them to specify their exact locations. This task is crucial for applications like autonomous driving, surveillance, and image analysis, where understanding the context and position of objects is essential.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_RetinaNet_ObjectDetection_mqeprgnq.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_retinanet_train_rpnekclt.py --train_dir /aicandy/datasets/aicandy_motorcycle_humukdiy --num_epochs 100 --batch_size 4 --model_path 'aicandy_output_ntroyvui/aicandy_model_retina_lgkrymnl.pth' 
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_retinanet_test_cliaskyp.py --image_path image_test.jpg --model_path 'aicandy_output_ntroyvui/aicandy_model_retina_lgkrymnl.pth' --class_list labels.txt --output_path aicandy_output_ntroyvui/image_out.jpg
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/su-dung-mang-neural-retinanet-vao-nhan-dien-doi-tuong).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️





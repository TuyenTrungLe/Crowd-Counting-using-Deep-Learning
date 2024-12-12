<h1 align="center">Crowd Counting Project</h1>

<p style="text-align: justify;">This project develops a deep learning model capable of estimating the number of people in an image.</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Setup and usage](#setup-and-usage)
5. [Model Description](#model-description)
6. [Authors](#authors)

## Overview

<p style="text-align: justify;">This project focuses on the development of a crowd counting model to estimate the number of people in a given image. The model is trained using the Mall dataset, which consists of 2000 frames taken from CCTV footage. Each frame has annotations regarding the number of people in the scene and their head locations. Also the whole project comes with a user-friendly Python Flask web interface.</p>

## Dataset

<p style="text-align: justify;">The Mall dataset is used for training and evaluation. It includes:</p>

- **2000 frames** from CCTV footage.
- Over **60,000** pedestrians annotated with **counts** and **head locations**.

<p style="text-align: justify;">You can download the dataset from <a href="https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html">here</a>.</p>

## Installation

<p style="text-align: justify;">To get started with the project, you need to have <a href="https://www.python.org/downloads/">Python 3.x</a> installed on your machine. If you have Python installed, skip this step.</p>

### Setup and usage

1. Clone this repository to your local machine

```bash
git clone https://github.com/tasitamas/crowd-counting-deep-learning-project.git
```

2. Navigate to the recently cloned folder

```bash
cd crowd-counting
```

3. Install the necessary packages from the requirements.txt

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

This will start a local server, and you can access the web app in your browser.

## Model Description

<p style="text-align: justify;">The model used for crowd counting is based on the CSRNet architecture as it is our baseline model. But we have a custom-built regression model that uses the CSRNet architecture as a feature extractor, plus has an additional dense layer, with this it estimates the number of people in an image.</p> 
<p style="text-align: justify;">The model is trained using the Mall dataset, which provides both the image and the annotations for the count of people and their head locations.</p>

## Authors

- Tuyen Trung Le
- Anett Johanik
- Tamás Tasi

# MNIST Digit Clustering Project

🧠 An interactive application demonstrating unsupervised learning on handwritten digits using **t-SNE dimensionality reduction** and **K-Means clustering**.

## Overview

This project visualizes how unsupervised machine learning algorithms can discover patterns in the MNIST dataset without explicit labels. The application uses:

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Reduces high-dimensional data to 2D for visualization
- **K-Means Clustering**: Groups similar digits into clusters
- **Streamlit**: Interactive web interface for exploring different parameters

![MNIST Clustering Result](docs\image1.png)
![MNIST Clustering Result](docs\image2.png)
![MNIST Clustering Result](docs\image3.png)
![MNIST Clustering Result](docs\image4.png)

## Features

- 📊 Interactive parameter controls (number of clusters, t-SNE perplexity, random state)
- 📈 Real-time evaluation metrics (accuracy, silhouette score)
- 🧩 Interactive 2D cluster visualization with Plotly
- 🎯 Post-clustering digit mapping to evaluate clustering quality

## Requirements

- Python 3.11 or higher
- Dependencies listed in `requirements.txt`

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using Poetry (if you have pyproject.toml)

```bash
poetry install
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Controls

- **Number of clusters (k)**: Adjust between 3 and 15 clusters
- **t-SNE Perplexity**: Tune between 5 and 50 to control local/global structure balance
- **Random State**: Set for reproducible results

## Dataset

The project uses the **Digits dataset** from scikit-learn:
- 1,797 samples of 8×8 handwritten digit images
- 10 classes (digits 0-9)
- 64 features (pixel values)

## Metrics Explained

- **Accuracy**: Percentage of correctly predicted digits after mapping clusters to their most common digit
- **Silhouette Score**: Measure of how similar points are to their own cluster compared to other clusters (ranges from -1 to 1, higher is better)

## Project Structure

```
MNIST-Clustering-Project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── pyproject.toml      # Python project metadata
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Technologies Used

- **Streamlit**: Web framework for data applications
- **scikit-learn**: Machine learning library
- **Plotly**: Interactive visualization
- **NumPy**: Numerical computing
- **Pillow**: Image processing

## License

This project is open source and available under the MIT License.

## Author

MNIST Clustering Project

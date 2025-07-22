# Space Weather Forecast using GNN
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/SyedMohammedSameer/SpaceWeatherForecast)

This project utilizes a Graph Neural Network (GNN) to forecast space weather, specifically predicting solar wind speed based on solar imagery. The model processes images, likely from the Solar Dynamics Observatory (SDO), by interpreting them as graphs where pixels are nodes. The relationships between these nodes are then learned to predict corresponding solar wind measurements.

## Model Architecture

The core of this project is the `BorisGraphNet`, a custom GNN implemented in PyTorch. Key features of the model include:

*   **Graph Representation**: Each solar image is treated as a graph, with individual pixels serving as nodes. The model processes a `128x128` resized version of the original images.
*   **Adjacency Matrix**: The model supports two distinct methods for defining the connections (edges) between nodes:
    1.  **Static Adjacency**: A pre-computed adjacency matrix based on the spatial distance between pixels. A Gaussian-like function determines the edge weights, creating a fixed graph structure for all images.
    2.  **Dynamic Edge Prediction**: A small, embedded neural network that learns the edge weights dynamically based on pixel coordinates. This allows the model to determine the most relevant pixel-to-pixel connections for the prediction task.
*   **Regression Output**: The GNN takes the graph representation of an image, aggregates features from neighboring nodes, and passes them through a final linear layer to output a single continuous value—the predicted solar wind speed.

## Dataset

The model is trained and evaluated using a combination of solar imagery and corresponding solar wind speed data.

*   **Training Data (2013, 2014, 2015, 2017)**:
    *   **Features**: Solar images loaded from a NumPy array (`train_images_2021_logn.npy`).
    *   **Targets**: Solar wind speed values sourced from `2013_sw_speed.csv`, `2014_sw_speed.csv`, `sw_speed_2015.csv`, and `solar_wind_latest_2017.csv`.

*   **Testing Data (2018)**:
    *   **Features**: Solar images loaded from a NumPy array (`img_2018.npy`).
    *   **Targets**: Solar wind speed values sourced from `sw_speed_2018.csv`.

**Note**: The dataset files are not included in this repository. You must acquire them and place them in the correct directory structure for the script to execute successfully.

## Prerequisites

To run this project, you need to install the required Python libraries.

```bash
pip install torch torchvision tensorflow pandas numpy opencv-python scikit-learn matplotlib
```

## Usage

The entire workflow is contained within the `Space Weather Prediction using GNN (1).ipynb` Jupyter Notebook.

1.  **Set up the Data Directory**:
    The notebook is configured to load data from a parent directory (`../`). You should place the notebook inside a folder and the required data files outside of it. Your directory structure should look like this:

    ```
    working_directory/
    ├── 2013_sw_speed.csv
    ├── 2014_sw_speed.csv
    ├── sw_speed_2015.csv
    ├── solar_wind_latest_2017.csv
    ├── sw_speed_2018.csv
    ├── train_images_2021_logn.npy
    ├── img_2018.npy
    └── SpaceWeatherForecast/
        └── Space Weather Prediction using GNN (1).ipynb
    ```

2.  **Run the Notebook**:
    *   Open `Space Weather Prediction using GNN (1).ipynb` in a Jupyter environment.
    *   Execute the cells sequentially.
    *   The script will load and preprocess the data, define the `BorisGraphNet` model, and initiate the training and evaluation process by calling the `myfun()` function in the final cell. The model uses Mean Squared Error (`mse_loss`) as its loss function for this regression task.

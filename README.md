# zebrafishBehaviour
# Abstract
The goal of the project is comparing different data-driven models for the behavior of juvenile zebrafish.
The models should be able to predict and simulate the individual motion of an animal reacting to its environment.
The environment in our case is another fish and the walls surrounding the arena.

We approximate the movement by a piece-wise linear model.
The wall forces are fit using a force-based approach.

We present multiple models for the social behavior, starting with a linear receptive field model.
This model is then enhanced by considering temporal dependencies and non-linear effects.
The final model is able to capture the entire trajectory distribution conditioned on the surroundings of the fish for each linear segment.
We achieve that using a mixture density recurrent neural network model.

# Structure
This repository is structured as follows:
- writing/
contains all slides, the project proposal and the report.

- src/
contains the entire source code of this project, including pre-processing, models and model analysis.
The script segmentation.py and binning.py are used to pre-process the data.
The files fit_wall.py, train_linear.py and train_mdn.py are used to fit the models.
Object oriented interfaces are contained in calovi.py (wall model) and social_models.py (social models).

- src/notebooks/
contain Jupyter notebooks that are used to create figures, contain explorative data analysis and evaluation of models

- src/mdn_model
contains a small library for the neural networks, written in pytorch.
This incldues the models and loss functions.

- data/
is an empty folder that is used to hold the (non-open) data.
The directory raw contains the raw data and the directory processed contains the processed data.

- models/


# Reproducing my results.
1. Put the raw data into the directory data/raw.
The raw data is hardcoded, the original data was named trial{id}_fish{fish_id}-anglefiltered.csv.
We only use the time, position and orientation columns of all csvs.
You might need to adapt the column names and file names if you change the input data.

2. Run the script src/segmentation.py to segment the motion into kicks and extract basic features.
A few things are hardcoded, such as thresholds for velocities, parameters of the velocity series smoothing and approximate fps of the input data.
You might need to adjust these parameters for new data.
To follow the segmentation procedure, look at the main method - every step is split into functions and it should be easy to follow the algorithms.

3. Run the script src/fit_wall.py to fit a force-based wall model.
TODO!

4. Run the script src/binning.py to extract the receptive field data.
Again, follow the main method.
We reshape the data, compute the local coordinate system and then discretize the space.

5. Run the scripts src/train_linear.py and src/train_mdn.py to train all models.

All of these steps save intermediate data in the directory data/processed and fitted models in models/.
It is important to run the scripts in this *exact* order.

After these steps are done, you can run the notebooks in src/notebooks/ to evaluate the model quality and create a visualization.

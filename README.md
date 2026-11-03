# Photonics Musings

A collection of personal explorations and computational projects in photonics, numerical modelling, and machine learning.

----------------------------------------------------------------------------------------------------------------

## Projects

### Machine Learning-Based Prediction of Fibre Chromatic Dispersion

*A Python-based study exploring the use of a neural network to predict chromatic dispersion in single-mode step-index optical fibres.*

#### Approach

**Physics-Based Fibre Modelling**

(a) Developed a simplified step-index optical fibre model to calculate the fundamental LP01 mode and its effective refractive index.

(b) Incorporated the wavelength-dependent refractive index of fused silica using the Sellmeier equation.

(c) Solved the LP01 characteristic equation numerically using `scipy.optimize.brentq` to obtain the modal propagation characteristics.

**Effective Index and Chromatic Dispersion**

(a) Calculated the effective refractive index as a function of wavelength, core radius, and core–cladding refractive-index difference.

(b) Used cubic spline interpolation to obtain the second-order wavelength derivative of the effective refractive index.

(c) Calculated chromatic dispersion from the wavelength dependence of the effective refractive index, accounting for both material and waveguide dispersion.

**Dataset Generation**

(a) Generated a dataset of 9,300 samples by varying three input parameters: fibre core radius (3.2–4.0 µm), core–cladding refractive-index difference (0.0025–0.0050), and operating wavelength (1300–1600 nm).

(b) Considered 300 distinct fibre designs, each evaluated at 31 wavelength points, producing 9,300 input-output samples.

(c) Used the physics-based model to calculate the corresponding chromatic dispersion values, which served as reference targets for machine-learning training.

**Neural Network Development**

(a) Implemented a multilayer perceptron (MLP) regression model using scikit-learn to predict chromatic dispersion from the three input parameters.

(b) Used a network architecture with three hidden layers (64, 64, and 32 neurons) and ReLU activation.

(c) Standardised the input features and target values using `StandardScaler` before training the network.

**Model Training and Evaluation**

(a) Split the dataset by fibre design into training (240 designs), validation (30 designs), and test (30 designs) sets to evaluate generalisation to previously unseen fibre configurations.

(b) Evaluated the trained model on 30 held-out fibre designs, comprising 930 samples across the 31 wavelength points for each design.

(c) Compared predicted dispersion values against the physics-model reference values using the coefficient of determination (R²) and mean absolute error (MAE).

#### Test Performance

| Metric | Result |
|---|---:|
| R² | 0.999893 |
| MAE | 0.0485 ps/(nm·km) |

PS: *The evaluation measures agreement with the physics-based model and does not represent validation against experimental measurements.*

----------------------------------------------------------------------------------------------------------------

## Libraries & Tools

**Programming Language:** Python

**Libraries:** NumPy, SciPy, scikit-learn, Matplotlib

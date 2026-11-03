# Photonics Musings

This will be collection of my personal explorations and computational projects in photonics, numerical modelling, and machine learning.

---

## Projects

### Machine Learning-Based Prediction of Fibre Chromatic Dispersion

Exploring neural network-based prediction of chromatic dispersion using a physics-based step-index fibre model.

#### Approach

**Setting Up the Fibre Model**

(a) Developed a simplified step-index optical fibre model to calculate the fundamental LP01 mode and its effective refractive index.

(b) Incorporated the wavelength-dependent refractive index of fused silica using the Sellmeier equation.

(c) Solved the LP01 characteristic equation numerically using `scipy.optimize.brentq` to obtain the modal propagation characteristics.

**Calculating Effective Index and Dispersion**

(a) Calculated the effective refractive index as a function of wavelength, core radius, and core–cladding refractive-index difference.

(b) Used cubic spline interpolation to obtain the second-order wavelength derivative of the effective refractive index.

(c) Calculated chromatic dispersion from the wavelength dependence of the effective refractive index, accounting for both material and waveguide dispersion.

**Building the Dataset**

(a) Generated a dataset of 9,300 samples by varying three input parameters: fibre core radius (3.2–4.0 µm), core–cladding refractive-index difference (0.0025–0.0050), and operating wavelength (1300–1600 nm).

(b) Considered 300 distinct fibre designs, each evaluated at 31 wavelength points, producing 9,300 input-output samples.

(c) Used the physics-based model to calculate the corresponding chromatic dispersion values, which served as reference targets for machine-learning training.

**Setting Up the Neural Network**

(a) Implemented a multilayer perceptron (MLP) regression model using `scikit-learn` to predict chromatic dispersion from the three input parameters.

(b) Used a network architecture with three hidden layers (64, 64, and 32 neurons) and ReLU activation.

(c) Standardised the input features and target values using `StandardScaler` before training the network.

**Training and Testing the Model**

(a) Split the dataset by fibre design into training (240 designs), validation (30 designs), and test (30 designs) sets to evaluate generalisation to previously unseen fibre configurations.

(b) Evaluated the trained model on 30 held-out fibre designs, comprising 930 samples across the 31 wavelength points for each design.

(c) Compared predicted dispersion values against the physics-model reference values using the coefficient of determination (R²) and mean absolute error (MAE).

#### Test Performance

Evaluated on 30 held-out fibre designs (930 samples), using the code in this repository with random seed 42:

| Metric | Result |
|---|---:|
| R² | 0.999854 |
| MAE | 0.0541 ps/(nm·km) |
| RMSE | 0.0745 ps/(nm·km) |

PS: The evaluation measures agreement with the physics-based model and does not represent validation against experimental measurements.

---

## Libraries & Tools

**Programming Language:** Python

**Libraries:** NumPy, SciPy, scikit-learn, Matplotlib

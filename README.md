# Photonics Musings

This will be my collection of personal explorations and computational projects in photonics, numerical modelling, and machine learning.

## Projects

### Machine Learning-Based Prediction of Fibre Chromatic Dispersion

A computational study combining a physics-based step-index optical fibre model with a multilayer perceptron (MLP) regression model to predict chromatic dispersion across different fibre designs.

The project includes numerical solution of the fundamental LP01 mode, Sellmeier-based refractive-index calculations, dataset generation, neural network training, and evaluation on unseen fibre designs.

**Test Performance**

On 30 held-out fibre designs (930 samples), the MLP achieved an R² of 0.999854 and an MAE of 0.0541 ps/(nm·km) against the physics-model reference values. These metrics measure agreement with the physics-based model, not experimental validation.

![Physics-model and MLP dispersion curves for four test designs](fibre-dispersion-ml/results/test_fibre_design_dispersion_comparison.png)

**Project repository:** [Fibre Dispersion Prediction Using Machine Learning](fibre-dispersion-ml/)

## Libraries & Tools

**Programming Language:** Python

**Libraries:** NumPy, SciPy, pandas, scikit-learn, Matplotlib, joblib

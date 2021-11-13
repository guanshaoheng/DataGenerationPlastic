# MCCDataGeneration
- This repository is used to generate the training data for our upcoming physics-constrained neural network training.
- The code is based on the modified Cam-Clay model.

## Yield function
<img src="doc/Equation/yieldfunc.gif" alt="" height="50" title="">

## Initial volum 
<img src="doc/Equation/InitialVolum.gif" height="50">

## Elastic modulus
<img src="doc/Equation/elasticModulus.gif" height="120">

## Gaussian process
- kernel function
<img src="doc/Equation/kernelFunction.gif" height="50">

- Gaussian function
<img src="doc/Equation/gaussianFunction.gif" height="50">

- Covariance Matrix
<img src="figSav/curlCoefComparation/CovariabceHeatMap_curl2.png" height="150">


- Gaussian random loading path

Gaussian random loading path          |  Deformation of the configuration
:-------------------------:|:------------------------------------:
<img src="figSav/curlCoefComparation/ConfiningPressureGP_curl2.png" height="200">  |  <img src="MCCData/animation/deformation_0.gif" height="200">



## Results of the simulation
Assuming that the loading will end up in the critical state.
<img src="figSav/MCCmodel-1.png" alt="MCC loading display" height="200" title="MCC loading display">
<img src="figSav/MCCmodel-2.png" alt="MCC loading display" height="200" title="MCC loading display">

## Size of the yield surface controlled by the hardening variable in the yield function
<img src="figSav/YieldSurface.svg" alt="MCC loading display" height="200" title="MCC loading display">

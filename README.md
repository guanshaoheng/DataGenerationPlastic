# Data Generation
- This repository is used to generate the training data for our upcoming physics-constrained neural network training.
- Guassian process is involved for random loading path generation.
- There 3 kinds of the elastoplastic model implemented here. 1. the model based on von-mises stress, iso-tropic exponential hardening and associated flow rule, 2. the modified Cam-Clay model, 3. the CSUH model.

## Gaussian process
These are used for random loading path generation.

- kernel function
<img src="doc/Equation/kernelFunction.gif" height="50">

- Gaussian function
<img src="doc/Equation/gaussianFunction.gif" height="50">

- Covariance Matrix
<img src="figSav/curlCoefComparation/CovariabceHeatMap_curl2.png" height="150">


- Gaussian random loading path

Gaussian random loading path          |  Deformation of the configuration
:-------------------------:|:-------------------------------:
<img src="figSav/curlCoefComparation/ConfiningPressureGP_curl2.png" height="200">  |  <img src="MCCData/animation/deformation_4.gif" height="200">


## MCC model (Modified Cam-Clay model)
### Yield function of the modified Cam-Clay model
<img src="doc/Equation/yieldfunc.gif" alt="" height="50" title="">

### Size of the yield surface controlled by the hardening variable in the yield function
<img src="figSav/YieldSurface.svg" alt="MCC loading display" height="200" title="MCC loading display">

### Initial volum 
<img src="doc/Equation/InitialVolum.gif" height="50">

### Elastic modulus
<img src="doc/Equation/elasticModulus.gif" height="120">

### Results of the Modified CamClay
Assuming that the loading will end up in the critical state.

Loading information         |  NCL & CSL
:-------------------------:|:-------------------------------:
<img src="figSav/MCCmodel-1.png" alt="MCC loading display" height="200" title="MCC loading display"> | <img src="figSav/MCCmodel-2.png" alt="MCC loading display" height="200" title="MCC loading display">

## CSUH model (Critical state unified hardening model)
There model is implemented under the guidance according to the paper (https://www.sciencedirect.com/science/article/pii/S0266352X19300576)

### Resualts of the undrained compression

e_0       |p0 |  Loading results
:-------------------------:|:-------------------------------:
 0.700 | p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.700_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
 0.747 |  p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.747_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
 0.794 |  p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.794_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
 0.841 |  p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.841_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
 0.888 |  p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.888_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
 0.935 |  p0=1000kPa|<img src="CSUHresults\undrained_voidratio/Toyoura_e0_0.935_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 


e_0     | p0    |  Loading results
:-------------------------:|:-------------------------------:
  0.833 |  p0=100 kPa|<img src="CSUHresults\undrained_pressure/Toyoura_e0_0.833_p0_100.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
  0.833 | p0=1000kPa|<img src="CSUHresults\undrained_pressure/Toyoura_e0_0.833_p0_1000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
  0.833 | p0=2000kPa|<img src="CSUHresults\undrained_pressure/Toyoura_e0_0.833_p0_2000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 
  0.833 |  p0=3000kPa|<img src="CSUHresults\undrained_pressure/Toyoura_e0_0.833_p0_3000.000kPa.png" alt="Undrained loading display" height="200" title="Undrained loading display"> 

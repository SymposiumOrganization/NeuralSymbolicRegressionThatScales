# NeuralSymbolicRegressionThatScales
Source code and Dataset creation for the paper "Neural Symbolic Regression That Scales" 

# How to use it
First you need to generate a dataset. Using the provided makefile is the easiest way to create it.
Define a variable in the console and export it to subprocesses by the command export NUM=${NumberOfEquationsYouWant}. 
NumberOfEquationsYouWant can be defined in two formats with K or M suffix. For instance

## Dataset Generation
Using the makefile is fastest way to create a datest. First, export the number of equations you would like to have with the command "export NUM=100000". Second, call make 
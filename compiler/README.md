# BYON: Bring Your Own Network Training

This tutorial is for REDsec's BYON training. The training consists of the following parts:
-   **REDsec CSV Netlist Generation**
-   **REDsec Code Generator**
-   **Larq Tensorflow Code Training**
-   **REDsec Weight Conversion**
-   **REDsec Inference Code**

This tutorial will walk you through each of these steps using a MNIST network. Outputs of each section are provided so that users can jump ahead and start anywhere along the toolchain. 

## CSV Netlist Generation

The CSV netlist generator is a a visual tool based in VBA. You will need to open up *REDsecNetlistGenerator.xlsm* inside the "compiler" directory, and enable Excel Macros on your device.

Upon opening the workbook, there will be a button to generate the netlist. A series of user forms will appear on the screen to enter in the neural network parameters. At the end, a seperate tab will be generated with the csv. 

To save as a csv stay in the tab with the designed neural network. Go to *File* -> *Export* -> *Change File Type* -> *Comma Separated Value*. It will prompt to save the file, and it will ask to confirm only the current tab should be exported.

Our `sample.csv` targets MNIST with three 1024-neuron hidden layers.

**Note:** We listed a [REDsec CSV netlist style guide](https://github.com/TrustworthyComputing/REDsec/blob/main/compiler/NetlistStyleGuide.md) to edit your network without the VBA tool.

## REDsec Code Generator

Our `compiler.py` program will generate the Larq TensorFlow training code and REDsec inference code automatically, based on the csv netlist. The output files will be in the nets/ directory.

To run the compiler, run the following, substituting *sample\_network* and *sample.csv* for your own files.
```	
python3 compiler.py sampleNetwork sample.csv
```
## Larq Tensorflow Code Training

Move to the directory with the new generated network.
```	
cd ../nets/sampleNetwork/sample
```
Here there will be a *sample.py* training file. By defualt, MNIST data was inserted into the network. The TODO block must be modified to insert the new target dataset. Then to train, run:
```	
python3 sample.py
```
## REDsec Weight Conversion

REDsec's optimized weight conversion and compression can be run as follows:

```
make weight_convert
```

## REDsec Inference

REDsec is now ready for inference! The file links in `main.cpp` can be edit to have the correct input data. The instructions for deploying a network can be found [here](https://github.com/TrustworthyComputing/REDsec).

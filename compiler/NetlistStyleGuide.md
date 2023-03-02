# REDsec csv Netlist style guide

Here we provide the parameters for designing your own netlist by hand, without the REDsecNetlistGenerator tool. 
Each row of the netlist represents a layer, and each column represents an operation. The compiler assumes that each operation is in the correct column. We also use ":" as a delimiter The format used is as follows:

##### Column 1: Input Data/Convolution/Fully Connected Layers

Input Data contains the input dimensions, including height, width and number of color channels. A fourth argument is the number of bits of each pixel (typically 8 bits). All arguments should be hardcoded integers.
```
input_size(pictureHeight : pictureWidth : pictureChannels : pictureBits)
```

Convolution data contains information about the depth (or output channels), convolution window (height and width), strides (height and width), padding type, and [ternary threshold\_value](https://docs.larq.dev/larq/api/quantizers/#stetern). 
Depth, window and stride are positive integers; padding type is either *same* or *valid*; and *ternaryThresholdValue* is between 0 and 1, with 0 representing a binary neural network and 0.05 being the default ternary neural network threshold. Convolution window and strides are surrounded by \{\}.
```
Convolution(convOutputDepth : {convWindowHeight : convWindowWidth} : {convStrideHeight : convStrideWidth} : same|valid : ternaryThresholdValue)
```

Fully connected data contain only output depth and [ternary threshold\_value](https://docs.larq.dev/larq/api/quantizers/#stetern). Both these parameters are the same as above:
```
FullyConnect(fcDepth : ternaryThresholdValue)
```

**Additional Rules**
The first layer must be an input layer, not a convolution or fully connected layer.

##### Column 2: MaxPool and SumPool Layers
Pooling can be selected for REDsec's neural network. Here we have the pooling window, pooling stride, and padding type as input arguments. The format is similar to convolution, where poolng window and stride must be positive integers and are surrounded by \{\}, and padding type can either be *same* or *valid*.
```
MaxPool({poolingWindowHeight : poolingWindowWidth} : {poolingStrideHeight : poolingStrideWidth} : same|valid)
SumPool({poolingWindowHeight : poolingWindowWidth} : {poolingStrideHeight : poolingStrideWidth} : same|valid)
```

**Additional Rules**
Max Pooling can only be used if the layer output uses a Sign activtion function.
Pooling can only be applied before flattening to a fully connected layers.

##### Column 3: BatchNorm
[BatchNorm](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) is recommended to be used by REDsec's neural network. Momentum and epsilon must be set for training. Larq recommends a [momentum](https://docs.larq.dev/larq/guides/bnn-optimization/) of 0.9 for binary neural networks, different from the TensorFlow default; the epsilon default is 0.001. Both should be values between 0 and 1. 
```
BNorm(momentum : eps)
```

**Additional Rules**
Batch Norm cannot be used in the last layer.

##### Column 4: Dropout
[Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) can be added, with a rate between 0 and 1.
```
Dropout(rate)
```

##### Column 5: Activation Function (Sign or ReLU)
An activation function must be added to every layer except the first layer (where it is optional) or the last layer (where it is not recommended). REDsec supports both [Sign](https://docs.larq.dev/larq/api/quantizers/#stesign) and [DoReFa ReLU](https://docs.larq.dev/larq/api/quantizers/#dorefa) activations.

For sign , no arguments are required:
```
Sign()
```

For ReLU, a single argument representing the number of output bits is chosen:
```
ReLU(outBits)
```
**Additional Rules**
Max Pooling can only be used if the layer output uses a Sign activtion function.

##### Column 6: Flatten
Before transistioning to fully connected layers, the Flatten() operation must be called.
```
Flatten()
```

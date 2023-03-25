# REDsec: Running Encrypted Discretized Neural Networks in Seconds  <a href="https://github.com/TrustworthyComputing/REDsec/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a> </h1>
REDsec is an end-to-end framework for encrypted
neural network inference using the CGGI cryptosystem. The [TFHE](https://github.com/tfhe/tfhe)
and [(RED)cuFHE](https://github.com/TrustworthyComputing/REDcuFHE) libraries are
utilized as cryptographic backends. A video presentation of REDsec is available [here](https://drive.google.com/file/d/19DTRfBerJ_CbMYzo2dTH1qdvWbZ0eecL/view?usp=sharing).

## Prerequisites
### Encrypted Inference with CPUs
Install [TFHE v1.1](https://github.com/tfhe/tfhe) with the packaged SPQLIOS-FMA FFT
engine.
### Encrypted Inference with NVIDIA GPUs
Install [(RED)cuFHE](https://github.com/TrustworthyComputing/REDcuFHE)

## Core Library Build Instructions
1. Clone the repo:  `git clone https://github.com/TrustworthyComputing/REDsec.git`
2. Navigate to library source directory: `cd REDsec/lib`
3. Build one (or more) of the desired library variants: `make ptxt && make cpu-encrypt && make gpu-encrypt` 

## Creating and Training Networks (CLOUD)
REDsec has a BYON framework. The tutorial and compiler are located [here](https://github.com/TrustworthyComputing/REDsec/compiler).

## Generating Keysets and Encryption/Decryption (CLIENT)
1. Navigate to client directory: `cd REDsec/client`
2. Generate FHE keypair by following the brief instructions printed by `make keygen-help`
3. Encrypt an image (in CSV format) by following instructions printed by `make
   encrypt-image-help`
4. After the inference procedure is run, follow the instructions printed by
   `make decrypt-image-help`

## Getting Started with Encrypted Inference (CLOUD)
1. Navigate to a network of your choice (e.g. `cd REDsec/nets/mnist/sign1024x1`)
2. _(Optional)_ Verify network accuracy: `make ptxt` 
3. Build and run one (or more) encrypted network variants : `make cpu-encrypt` for TFHE
   (CPU) backend and `make gpu-encrypt` for (RED)cuFHE (GPU) backend

## Cite this work
This library was introduced in the REDsec paper, which presents a framework for
privacy-preserving neural network inference and will appear in the NDSS
Symposium 2023 ([Cryptology ePrint
Archive](https://eprint.iacr.org/2021/1100.pdf)):
```
@inproceedings{folkerts2021redsec,
    author       = {Lars Folkerts and Charles Gouert and Nektarios Georgios Tsoutsos},
    title        = {{REDsec: Running Encrypted Discretized Neural Networks in Seconds}},
    booktitle = {{Network and Distributed System Security Symposium (NDSS)}},
    pages        = {1--17},
    year         = {2023},
}
```

<p align="center">
    <img src="./logos/twc.png" height="20%" width="20%">
</p>
<h4 align="center">Trustworthy Computing Group</h4>

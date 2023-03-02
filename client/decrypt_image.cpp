#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <vector>

using namespace std;

int num_classes = 0;
int num_bits = 0;
int msg_space = 0;

int pow_int(int base, int exponent) {
    if (exponent == 0)
        return 1;

    int result = pow_int(base, exponent / 2);
    result *= result;

    if (exponent & 1)
            result *= base;

    return result;
}    

int main(int argc, char** argv) {

  if (argc != 2) {
    cout << "Usage: ./decrypt.out [DATA_FORMAT]" << endl;
    return -1;
  }

  string data_fmt = argv[1];
  if (data_fmt  == "MNIST" || data_fmt == "CIFAR-10") {
    num_classes = 10;
  }
  else if (data_fmt == "ImageNet") {
    num_classes = 1000;
  }
  else {
    cout << "Invalid data format!" << endl;
    return -1;
  }

  ifstream myfile("bitsize.data");
  int msg_bits = 0;
  if (myfile.is_open()) {
    myfile >> msg_bits;
    myfile.close();
  }
  if (msg_bits < 1) {
    cout << "Invalid bitsize, re-run inference procedure!" << endl;
    return 0;
  }
  msg_space = pow_int(2, msg_bits);

  FILE* secret_key = fopen("secret.key", "rb");
  TFheGateBootstrappingSecretKeySet* key = new_tfheGateBootstrappingSecretKeySet_fromFile(secret_key);
  fclose(secret_key);

  // get embedded params
  const TFheGateBootstrappingParameterSet* params = key->params;

  LweSample* e_labels = new_LweSample_array(num_classes, params->in_out_params);

  FILE* inference_output = fopen("network_output.ctxt","rb");
  vector<int> labels;
  for (int i = 0; i < num_classes; i++) {
      import_gate_bootstrapping_ciphertext_fromFile(inference_output, &e_labels[i], params);
      auto tmp = modSwitchFromTorus32(lweSymDecrypt(e_labels + i, key->lwe_key, msg_space),msg_space);
      if (tmp > (msg_space >> 1)) {
        labels.push_back(tmp - msg_space);
      }
      else {
        labels.push_back(tmp);
      }
  }
  fclose(inference_output);

  int bestScoreIdx = std::max_element(labels.begin(),labels.end()) - labels.begin();
  cout << "Classification Result: " << bestScoreIdx << endl;
  
  delete_gate_bootstrapping_secret_keyset(key);
  delete_gate_bootstrapping_ciphertext_array(num_classes, e_labels);
}

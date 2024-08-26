#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <vector>
#include "REDcuFHE/redcufhe_gpu.cuh"

using namespace std;
using namespace redcufhe;

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
  if (data_fmt == "MNIST" || data_fmt == "CIFAR-10") {
    num_classes = 10;
  }
  else if (data_fmt == "ImageNet") {
    num_classes = 1000;
  }
  else {
    cout << "Invalid data format!" << endl;
    return -1;
  }

  msg_space = 4096;
  
  PriKey secret_key;
  ReadPriKeyFromFile(secret_key, "secret.key");

  Ctxt* e_labels = new Ctxt[num_classes];
  vector<int> pt_labels(num_classes);
  std::ifstream input_file("network_output.ctxt");
  for (int i = 0; i < num_classes; i++) {
    ReadCtxtFromFileRed(e_labels[i], input_file);
    DecryptIntRed(pt_labels[i], e_labels[i], msg_space, secret_key);
    if (pt_labels[i] > (msg_space/2)) {
      pt_labels[i] -= msg_space; 
    }
  }
  input_file.close();

  int bestScoreIdx = std::max_element(pt_labels.begin(),pt_labels.end()) - pt_labels.begin();
  cout << "Classification Result: " << bestScoreIdx << endl;

  delete [] e_labels;
}

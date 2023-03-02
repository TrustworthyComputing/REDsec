#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "REDcuFHE/redcufhe_gpu.cuh"

using namespace std;
using namespace redcufhe;

int main(int argc, char** argv) {

  // import private key
  PriKey secret_key;
  ReadPriKeyFromFile(secret_key, "secret.key");

  string image;
  ifstream im_file (argv[1]);
  getline(im_file, image);

  string delim = ",";
  size_t pos = 0;
  string curr_val;
  uint32_t count = 0;

  string label = "0";
  string str_length = "0";
  string str_width = "0";
  string str_channels = "0";

  // extract parameters from file preamble
  while ((pos = image.find(delim)) != string::npos) {
      curr_val = image.substr(0, pos);
      image.erase(0, pos + delim.length());
      if (count == 0) {
          label = curr_val;
          count++;
          continue;
      }
      else if (count == 1) {
          str_length = curr_val;
          count++;
          continue;
      }
      else if (count == 2) {
          str_width = curr_val;
          count++;
          continue;
      }
      else if (count == 3) {
          str_channels = curr_val;
          count++;
          break;
      }
  }

  int length = stoi(str_length, nullptr, 10);
  int width = stoi(str_width, nullptr, 10);
  int channels = stoi(str_channels, nullptr, 10);
  int num_ctxts = length * width * channels * 8;

  Ctxt* e_image = new Ctxt[num_ctxts];
  count = 0;
  
  Ptxt* pt = new Ptxt[8];
  while ((pos = image.find(delim)) != string::npos) {
      curr_val = image.substr(0, pos);
      image.erase(0, pos + delim.length());
      int ptxt_val = stoi(curr_val, nullptr, 10);
      for (int i=count; i<count+8; i++) {
        pt[i-count].message_ = (ptxt_val>>i)&1;
        Encrypt(e_image[i], pt[i-count], secret_key);
        WriteCtxtToFileRed(e_image[i], "image.ctxt");
      }
      count += 8;
  }

  delete [] pt;
  delete [] e_image;
}

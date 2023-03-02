#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define SECALPHA pow(2., -25)

using namespace std;

int main(int argc, char** argv) {

  const double alpha = SECALPHA;
  // import private key
  FILE* secret_key = fopen("secret.key", "rb");
  TFheGateBootstrappingSecretKeySet* key = new_tfheGateBootstrappingSecretKeySet_fromFile(secret_key);
  fclose(secret_key);

  // get embedded params
  const TFheGateBootstrappingParameterSet* params = key->params;

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
  int num_ctxts = length * width * channels;

  LweSample* e_image = new_LweSample_array(num_ctxts, params->in_out_params);
  int j = 0;

  while ((pos = image.find(delim)) != string::npos) {
		curr_val = image.substr(0, pos);
		image.erase(0, pos + delim.length());
		int ptxt_val = stoi(curr_val, nullptr, 10)*2-255;
		lweSymEncrypt(&e_image[j], modSwitchToTorus32(ptxt_val, 2048), SECALPHA, key->lwe_key);
		// lweNoiselessTrivial(&e_image[j], modSwitchToTorus32(ptxt_val, 2048), params->in_out_params);
		j++;
  }

  FILE* answer_data = fopen("image.ctxt","wb");
  for (int i=0;i<num_ctxts;i++)
		export_gate_bootstrapping_ciphertext_toFile(answer_data, &e_image[i], params);
  fclose(answer_data);
}

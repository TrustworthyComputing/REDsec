#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include "../../../lib/GPU/Layer.cuh"
#include "net.cuh"
#include "../mnist.h"
#include <ctime>
#include <ratio>
#include <chrono>

#include "REDcuFHE/redcufhe_gpu.cuh"

/*********************************** Defines ************************************/
#define CSV_FIELD_LEN 4 //3digit number with a comma seperator
#define NUM_SAMPLES    ((uint32_t)100)
#define UPDATES     ((uint32_t)(NUM_SAMPLES/10.0+.999))
/****************************** Structs And Enums *******************************/
typedef eMnist eResult ;
/**************************** Function Declarations *****************************/
void reset_dim(tDimensions* dim) ;

using namespace redcufhe;
using namespace std::chrono;

int main(void)
{
    //dimensions
    tDimensions indim ;
    tDimensions outdim ;
    //create network
    cudaSetDevice(0);
    print_status("Instantiating network architecture...\n");
    HeBNN* network = new HeBNN() ;
    network->get_in_dims(&indim) ;
    network->get_out_dims(&outdim) ;

    //Results
    size_t in_size = indim.hw.h * indim.hw.w * indim.in_dep;
    tMultiBitPacked* nn_data;
    mbit_calloc_global(&nn_data, in_size, 8);
    tMultiBitPacked* nn_result ;
    mbit_calloc_global(&nn_result, 10, 1);

    // Read encrypted image
    for (int k = 0; k < NUM_GPUS; k++) {
      cudaSetDevice(k);
      std::ifstream input_file("../../../client/image.ctxt");
      for(uint32_t i = 0; i < in_size; i++) {
        for (uint8_t j = 0; j < 8; j++) {
          ReadCtxtFromFileRed(nn_data->enc_segs[k][i].ctxt[j], input_file);
        }
      }
      input_file.close();
    }

    printf("Running network...\n");
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    nn_result = network->run(nn_data);
    Synchronize();
    CuCheckError();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Inference Time: " << time_span.count() << " seconds\n";
    CleanUp();
}

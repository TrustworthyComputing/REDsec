#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../mnist.h"

#if defined(GPU_ENC)
#include "../../../lib/GPU/Layer.cuh"
#include "net.cuh"
#include "REDcuFHE/redcufhe_gpu.cuh"
#include <ctime>
#include <ratio>
#include <chrono>
using namespace redcufhe;
using namespace std::chrono;
#else
#include "../../../lib/Layer.h"
#include "net.h"
#endif


/*********************************** Defines ************************************/
#define CSV_FIELD_LEN 4 //3digit number with a comma seperator
#define NUM_SAMPLES    ((uint32_t)1)
#define UPDATES     ((uint32_t)(NUM_SAMPLES/10.0+.999))
/****************************** Structs And Enums *******************************/
typedef eMnist eResult ;
/**************************** Function Declarations *****************************/
#ifndef ENCRYPTED
eResult findResult(tFixedPoint* data, tDimensions* dim) ;
static tFixedPoint* convert_line(char* line, tDimensions* dim, eResult* p_label) ;
#endif
void reset_dim(tDimensions* dim) ;

int main(void)
{
    //dimensions
    tDimensions indim ;
    tDimensions outdim ;

    //create network
    #if defined(GPU_ENC)
    cudaSetDevice(0);
    #endif 
    print_status("Instantiating network architecture...\n");
    HeBNN* network = new HeBNN() ;
    network->get_in_dims(&indim) ;
    network->get_out_dims(&outdim) ;

    //csv input
    FILE *fd_image = NULL ;

    //Results
    #if defined(GPU_ENC)
    size_t in_size = indim.hw.h * indim.hw.w * indim.in_dep;
    tMultiBitPacked* nn_data;
    mbit_calloc_global(&nn_data, in_size, 8);
    tMultiBitPacked* nn_result ;
    mbit_calloc_global(&nn_result, outdim.in_dep, 1);

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
    exit(0);
    #elif defined(ENCRYPTED)
    size_t in_size = indim.hw.h * indim.hw.w * indim.in_dep ;
    tMultiBit* nn_data = new tMultiBit[in_size];
    for (uint32_t i = 0; i < in_size; i++) {
      nn_data[i].size = 1;
      nn_data[i].ctxt = new_gate_bootstrapping_ciphertext_array(nn_data[i].size, network->bk->params);
    }
    tFixedPoint* nn_result ;
    #else
    size_t line_size = (CSV_FIELD_LEN * (indim.hw.h) * (indim.hw.w) * (indim.in_dep)) ;
    char* file_line = (char*) malloc(line_size) ;
    tFixedPoint* nn_data ;
    uint32_t image_i = 0 ;
    uint32_t correct = 0 ;
    eResult label = RESULT_ERROR ;
    eResult result = RESULT_ERROR ;
    tFixedPoint* nn_result ;
    #endif

    //Open File
    #if defined(ENCRYPTED)
    fd_image = fopen("../../../client/image.ctxt", "rb");
    #elif !defined(GPU_ENC)
    fd_image = fopen("../mnist_test.csv", "r");
    #endif

    #if !defined(GPU_ENC)
    if(fd_image == NULL)
    {
        printf("Bad Sample File. Exiting...\r\n") ;
        return -1 ;
    }
    #endif

    #ifdef ENCRYPTED //only one image
    for(uint32_t i = 0; i < in_size; i++) {
      import_gate_bootstrapping_ciphertext_fromFile(fd_image, &nn_data[i].ctxt[0], network->bk->params);
    }
    printf("Running network...\n");
    nn_result = network->run(nn_data);
    FILE* output_ctxts = fopen("../../../client/network_output.ctxt", "wb");

    // output result to file
    for (uint8_t i = 0; i < outdim.in_dep; i++) {
      export_gate_bootstrapping_ciphertext_toFile(output_ctxts, &nn_result[i].ctxt[0], network->bk->params);
    }

    printf("Result ctxts loaded into network_output.ctxt.\n");
    fclose(output_ctxts);

    // clean up ctxt
    for (uint32_t i = 0; i < outdim.in_dep; i++) {
      delete_gate_bootstrapping_ciphertext_array(nn_result[i].size, nn_result[i].ctxt);
    }
    free(nn_result);
    #else
    //Run network on images
    for(image_i = 0 ; image_i < NUM_SAMPLES ; image_i++)
    {
      label = RESULT_ERROR ;
      result = RESULT_ERROR ;
      //get line
      file_line = fgets(file_line, line_size, fd_image) ;
      if(*file_line < '0' || *file_line>'9'){ continue ; }
      //convert data to appropriate struct
      nn_data = convert_line(file_line, &indim, &label) ;
      //Run NN
      nn_result = (tFixedPoint*) network->run(nn_data) ;
      //Get result
      result = findResult(nn_result, &outdim) ;
      if(result == label){ correct++ ; }
      if((image_i % UPDATES)==0)
      {
        printf("correct: %4d\timage_i: %4d\tLabel: %d\tPrediction: %d\r\n", correct, image_i, label, result) ;
      }
      free(nn_result) ;
    }
    printf("Correct: %2f%%\r\n", ((float)correct*100.0)/image_i) ;
    #endif
    fclose(fd_image) ;

    #ifdef ENCRYPTED
    delete_gate_bootstrapping_cloud_keyset(network->bk);
    #endif
}

#if !defined(ENCRYPTED)
eResult findResult(tFixedPoint* data, tDimensions* dim)
{
  tFixedPoint val = 0 ;
  eResult res = RESULT_ERROR ;

  if(data != NULL)
  {
    val = data[0];
    res = CAT_0 ;
    for(uint16_t image_i = (int) CAT_0 ; image_i < (dim->in_dep) ; image_i++)
    {
      if(data[image_i] > val)
      {
        res = (eResult) image_i ;
        val = data[image_i] ;
      }
    }
  }
  return res ;
}

//TODO: THis is the function preprocessing
static tFixedPoint* convert_line(char* line, tDimensions* dim, eResult* p_label)
{
  uint32_t len = get_size(&(dim->hw), dim->in_dep, SIZE_EMPTY) ;
  tFixedPoint* output = (tFixedPoint*) calloc(len, sizeof(tFixedPoint)) ;
  char* tok = strtok(line, ",") ; //first number is the label
  *p_label = (eResult) atoi(tok) ;
  for (uint32_t i = 0 ; i < len ; i++)
  {
    tok = strtok(NULL, ",\n") ;
    if((tok != NULL) && (*tok != '\0'))
    {
      output[i] = (tFixedPoint) (2*atoi(tok)-255) ;
    }
  }
  return output ;
}
#endif

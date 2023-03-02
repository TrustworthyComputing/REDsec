/*********************************************************************************
*    File:        net.cpp
*    Desc:        Contains all the QNN layers
*    Authors:    Lars Folkerts
*    Notes:        None
*********************************************************************************/
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Layer.h"
#include "BinLayer.h"
#include "IntLayer.h"

#include "net.h"
#include "../mnist.h"

/*********************************** Defines ************************************/
#define DEPTH_1 ((uint16_t)1024)
#define OUT_DEPTH ((uint16_t)10)

#define RES_BITS 1
#define RESOLUTION (1<<RES_BITS)
#define CONV_WIND 3
#define POOL_WIND 2

HeBNN::HeBNN(FILE* in_file, bool b_prep)
{
    HeBNN::init(in_file, false) ;
}

HeBNN::HeBNN()
{
    //open up input file
    FILE *fd = fopen("var_prep.dat", "r");
    if(fd == NULL)
    {
        printf("Bad Weights File. Exiting...\r\n") ;
        return ;
    }
    HeBNN::init(fd, false) ;
    fclose(fd) ;
}

void HeBNN::init(FILE* in_file, bool b_prep)
{
    tDimensions* p_dim ;
    tDimensions lay_dim ;

    // read eval key
    #ifdef ENCRYPTED
    FILE* cloud_key = fopen("../../../client/eval.key","rb");
    bk = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
    fclose(cloud_key);
    #else
    bk = NULL;
    #endif

    //make standard convolution and pooling windows
    tNetParams params ;
    params.conv.window.h = 1 ; //CONV_WIND ;
    params.conv.window.w = 1 ; //CONV_WIND ;
    params.conv.stride.h = 1 ;
    params.conv.stride.w = 1 ;

    params.conv.same_pad = false ;
    params.conv.tern_thresh = 0.05 ;

    params.pool.window.h = POOL_WIND ;
    params.pool.window.w = POOL_WIND ;
    params.pool.stride.h = POOL_WIND ;
    params.pool.stride.w = POOL_WIND ;
    
    params.pool.same_pad = false ;

    params.bnorm.use_scale = false ;
    params.bnorm.eps = 0.001 ;

    params.quant.shift_bits = 1 ;

    params.version = 2 ;

    /* Create the model */
    //Layer 0: Reduce to 14x14 binary (preprocess)
    params.e_bias = E_NO_BIAS ;
    first_layer = new IntLayer(E_NO_CONV, SIZE_EMPTY, E_SUMPOOL, E_ACTIVATION_SIGN, &params, bk) ;
    //Mid-Layer neurons
    params.e_bias = E_BNORM ;
    layers.push_back(new BinLayer(E_FC, DEPTH_1, E_NO_POOL, E_ACTIVATION_SIGN, &params, bk)) ;
    //Mid-Layer neurons
    layers.push_back(new BinLayer(E_FC, DEPTH_1, E_NO_POOL, E_ACTIVATION_SIGN, &params, bk)) ;
    //Mid-Layer neurons
    layers.push_back(new BinLayer(E_FC, DEPTH_1, E_NO_POOL, E_ACTIVATION_SIGN, &params, bk)) ; 
    //Final FC Layer
    params.e_bias = E_NO_BIAS ;
    last_layer = new BinLayer(E_FC_FINAL, OUT_DEPTH, E_NO_POOL, E_ACTIVATION_NONE, &params, bk) ;

    //extract weights and initialize the dimensions of the model
    lay_dim.hw.h=PIC_SIZE ;
    lay_dim.hw.w=PIC_SIZE ;
    lay_dim.in_dep = IN_CHANNELS ;

    lay_dim.in_bits = PIXEL_BITS+1 ;
    lay_dim.out_bits = SINGLE_BIT ;
    lay_dim.filter_bits = SINGLE_BIT ;
    lay_dim.bias_bits = SINGLE_BIT ;
    lay_dim.up_bound = 255*2 ;
    lay_dim.scale = 255 ;

    p_dim = first_layer->prep(in_file, &lay_dim) ;
    for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
        p_dim = (*layer)->prep(in_file, p_dim) ;
    }
    p_dim = last_layer->prep(in_file, p_dim) ;
}


#if defined(ENCRYPTED)
tMultiBit* HeBNN::run(tMultiBit* in_data)
#else
tMultiBit* HeBNN::run(tFixedPoint* in_data)
#endif
{
    tBit* bdata ;
    tMultiBit* mbdata ;
    bdata =  (tBit*) first_layer->execute(in_data);
    for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
        bdata =  (tBit*) (*layer)->execute(bdata) ;
    }
    mbdata = (tMultiBit*)last_layer->execute(bdata) ;
    return (tMultiBit*)mbdata ;
}

void HeBNN::get_in_dims(tDimensions* in)
{
    if(in != NULL)
    {
        memcpy(in, &(first_layer->in_dim), sizeof(*in)) ;
    }
}

void HeBNN::get_out_dims(tDimensions* out)
{
    if(out != NULL)
    {
        memcpy(out, &(last_layer->out_dim), sizeof(*out)) ;
    }
}

#ifdef _WEIGHT_CONVERT_
void HeBNN::export_weights(FILE* out_file)
{
    print_status("Begin Export\r\n") ;
    first_layer->export_weights(out_file);
    for(auto layer = layers.begin(); layer != layers.end(); ++layer)
    {
        (*layer)->export_weights(out_file);
    }
    last_layer->export_weights(out_file);
    print_status("Exported last layer\r\n") ;

    return ;
}
#else
void export_weights(FILE* out_file){ printf("Weight Convert not defined\r\n") ; }
#endif

/**************************************************************************************************
* FILENAME :        IntLayer.cpp
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file pieces together the integer Convolution/Fully Connected, Batch Norm, Pooling and 
*       Activation functions in the optimal order for TFHE REDsec. 
*
* NOTES :     
* AUTHOR :    Lars Folkerts
* START DATE :    16 Aug 20
*************************************************************************************************/

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstring>
#include "Layer.h"
#include "IntFunc.h"
#include "BinFunc.h"
#include "IntLayer.h"

/*********************************************************************************
*    Func:      IntLayer::IntLayer
*    Desc:      Creates a Integer Layer Object
*    Inputs:    eConvType - Type of Convolution
*               uint16_t - ouput depth
*               ePoolType - type of Pooling
*               eQuantType - type of activations
*               tNetParams* - input network parameters
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key
*                       (or NULL if unencypted)
*    Return:    IntLayer Object
*    Notes:    
*********************************************************************************/
IntLayer::IntLayer(eConvType ec, uint16_t dep, ePoolType ep, eQuantType eq, 
                   tNetParams* np, TFheGateBootstrappingCloudKeySet* in_bk)
{
    assert((ec < NUM_CONVS) && (np->e_bias < NUM_BIASES) && (ep < NUM_POOLS)) ;
    assert((np != NULL)) ;

    tNetParams net_param ;
    memcpy(&net_param, np, sizeof(tNetParams)) ;
    e_conv = ec ;
    conv_depth = dep ;
    e_bias = np->e_bias ;
    e_pool = ep ;
    e_activation = eq ;
    bk = in_bk ;
    p_bias = NULL ; //to be allocated
    p_slope = NULL ; //may be allocated

    //set defaults from previous versions
    set_version(net_param.version, &net_param) ; 
    
    if((e_conv == E_FC_FINAL) || (e_conv == E_FC))
    {
        net_param.conv.window.h = 1 ;
        net_param.conv.window.w = 1 ;
        net_param.conv.same_pad = true ;
    }

    //Layers - Convolution
    lconv = new IntFunc::Convolution(conv_depth, &(net_param.conv)) ;
   	//Pooling
    if(e_pool == E_SUMPOOL){ lpool.s = new IntFunc::SumPooling(&(net_param.pool)) ; }
    else if(e_pool == E_MAXPOOL)
    {
	    assert(eq == E_ACTIVATION_SIGN) ;
	    lpool.m = new BinFunc::MaxPooling(&(net_param.pool)) ; 
    }
    //Quantizing
    if((e_activation == E_ACTIVATION_SIGN)){ net_param.quant.shift_bits = 1 ; }
    else if(e_activation == E_ACTIVATION_NONE){ net_param.quant.shift_bits = 0 ; }
    lquant = new IntFunc::Quantize(&(net_param.quant)) ; 
#ifdef _WEIGHT_CONVERT_
    //Batch Norm
    if(e_bias == E_BNORM){ bnorm = new IntFunc::BatchNorm(&(net_param.bnorm)) ; }
    else{ bnorm = NULL ; }
#endif
}

/*********************************************************************************
*    Func:      IntLayer::prep
*    Desc:      Prepares the integer layer, reading weights and allocating memory
*    Inputs:    FILE* - weights file
*               tDimensions* - input dimensions
*    Return:    tDimensions* - output dimensions
*    Notes:     Input weights file is compressed for evaluation or TensorFlow for
*               weight convert
*********************************************************************************/
tDimensions* IntLayer::prep(FILE* fd, tDimensions* dim)
{
    //allocate bias array
    p_bias = mbit_calloc(conv_depth, FIXEDPOINT_BITS, bk) ;
    #ifndef ENCRYPTED
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM)) {    p_slope = mbit_calloc(conv_depth, FIXEDPOINT_BITS, bk) ; }
    #else
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM)){ p_slope = (uint32_t*) calloc(conv_depth, sizeof(uint32_t)) ; }
    #endif

    //initialize arguments
    tActParams pact ;
    pact.d = dim ;
    //save/calculate dimensions and extract weights
    memcpy(&in_dim, pact.d, sizeof(in_dim)) ;
    pact.d = (tDimensions*) IntLayer::run(E_PREP, &pact, fd) ;
    memcpy(&out_dim, pact.d, sizeof(out_dim)) ;

#ifdef _WEIGHT_CONVERT_
    tActParams dummy ;
    void* dp = IntLayer::run(E_PREP_BIAS, &dummy, fd) ;
#endif
    return pact.d ;
}

/*********************************************************************************
*    Func:      IntLayer::execute
*    Desc:      Runs all operations on the layer
*    Inputs:    tMultiBit* - input to layer
*    Return:    void* - layer outputs (either tBit* or tMultiBit*)
*    Notes:     None
*********************************************************************************/
#if defined(ENCRYPTED)
void* IntLayer::execute(tMultiBit* pfp_in)
#else
void* IntLayer::execute(tFixedPoint* pfp_in)
#endif
{
    tActParams pact ;
    pact.fp = pfp_in ;
    return (tBit*) IntLayer::run(E_EXEC, &pact, NULL) ;
}

void IntLayer::export_weights(FILE* fd_bin)
{
    tActParams dummy ;
    void* d = IntLayer::run(E_EXPORT, &dummy, fd_bin) ;
}

/*********************************************************************************
*    Func:      IntLayer::run
*    Desc:      Helper Function to prep, execute, export weights
*    Inputs:    eAction - either prepare or execute the laye
*               tActParams - input parameters (either tDimiensions* or tBit*)
*               FILE* - input weights file (weight convert prep),
*                       output weights file (weight convert execute),
*                       or NULL (running neural network)
*    Return:    void* - layer outputs (either tDimensions*, tBit*, or tMultiBit*)
*    Notes:     None
*********************************************************************************/
void* IntLayer::run(eAction e_act, tActParams* ap, FILE* fd)
{
    assert(ap != NULL) ;
    tFixedPoint* fpdata  = ap->fp ;
    tBit* bdata ;

    tDimensions* dim = ap->d ;

    //First is convolution
    switch(e_conv)
    {
        case E_FC_FINAL:
        case E_FC:
        {
            if(e_act == E_PREP)
            {
                //flatten
                dim->in_dep *= dim->hw.h * dim->hw.w ;
                dim->hw.h = 1 ;
                dim->hw.w = 1 ;
             }
        }
	    //fall through
        case E_CONV:
        {
            if(e_act == E_PREP){ dim = lconv->prep(fd, dim, bk) ; }
            else if(e_act == E_EXEC){ fpdata = lconv->execute(fpdata) ; }
            //weight prep/export
            else if(e_act == E_PREP_BIAS){ lconv->extract_bias(fd, p_bias, e_bias) ; }
            else if(e_act == E_EXPORT){ lconv->export_weights(fd) ; }
        }
        break ;
	    case E_NO_CONV:
	    break ;
        default: { assert(0) ;}
        break ;
    }
#ifdef _WEIGHT_CONVERT_
    if(e_bias == E_BNORM)
    {
        if(e_act == E_PREP){ dim = bnorm->prep(dim, in_dim.in_bits) ; }
        if(e_act == E_PREP_BIAS){ bnorm->extract_bias(fd, p_bias, p_slope) ; }
    }
#endif

    //Sumpooling
    if(e_pool == E_SUMPOOL)
    {
        if(e_act == E_PREP){ dim = lpool.s->prep(dim, bk) ; }
        else if(e_act == E_EXEC){ fpdata = lpool.s->execute(fpdata) ; }
        else if(e_act == E_PREP_BIAS){ lpool.s->extract_bias(p_bias) ; }
    }
    
    //quantize
    if(e_act == E_PREP){
        dim = lquant->prep(fd, dim, p_bias, p_slope, bk) ;
    }
    else if(e_act == E_EXEC)
    {
        if(e_activation == E_ACTIVATION_SIGN){bdata = lquant->execute(fpdata, p_bias) ; }
	    else if(e_activation == E_ACTIVATION_NONE){ fpdata = lquant->add_bias(fpdata, p_bias) ; }
	    else if(e_activation == E_ACTIVATION_RELU){ fpdata = lquant->relu_shift(fpdata, p_bias, p_slope) ; }
    }
    #ifndef ENCRYPTED
    else if(e_act == E_PREP_BIAS){ lquant->extract_bias(p_bias, p_slope) ; } 
    else if(e_act == E_EXPORT){ lquant->export_weights(fd, p_bias, p_slope) ; }
    #endif
    //Max Pooling
    if(e_conv != E_FC_FINAL)
    {
        if(e_pool == E_MAXPOOL)
        {
            if(e_activation != E_ACTIVATION_SIGN){ printf("Unexpected MaxPooling on Integers\r\n") ; }
            else if(e_act == E_PREP){ dim = lpool.m->prep(dim, bk) ; }
            else if(e_act == E_EXEC){ bdata = lpool.m->execute(bdata) ; }
        }
    }

    if(e_act == E_PREP){ return (void*) dim ; }
    else if(e_activation == E_ACTIVATION_SIGN && e_act == E_EXEC){ return (void*) bdata ; }
    else if(e_act == E_EXEC){ return (void*) fpdata ; }
    return (void*) NULL ;
}

//Version changes - reverse compatibility
/*********************************************************************************
*    Func:      IntLayer::set_version
*    Desc:      Modifies arguments for reverse compatibility
*    Inputs:    uint8_t - version
*               tNetParams* - parameters to be modified
*    Return:    None
*    Notes:     None
*********************************************************************************/
void IntLayer::set_version(uint8_t v, tNetParams* net)
{
    if(v<1)
    {
        net->conv.stride.h = 1 ;
        net->conv.stride.w = 1 ;
        net->pool.stride.h = 0 ; 
        net->pool.stride.w = 0 ;
    }
}

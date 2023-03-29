/**************************************************************************************************
* FILENAME :        BinLayer.cpp
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file pieces together the binary Convolution/Fully Connected, Batch Norm, Pooling and 
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
#include "BinFunc.h"
#include "BinLayer.h"

/*********************************************************************************
*    Func:      BinLayer::BinLayer
*    Desc:      Creates a Binary Layer Object
*    Inputs:    eConvType - Type of Convolution
*               uint16_t - ouput depth
*               ePoolType - type of Pooling
*               eQuantType - type of activations
*               tNetParams* - input network parameters
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key 
*               	(or NULL if unencypted)
*    Return:    BinLayer Object
*    Notes:     Assumes inputs are -1 (represented as bit 0) 
*    		or +1 (represented by bit 1)
*********************************************************************************/
BinLayer::BinLayer(eConvType ec, uint16_t dep, ePoolType ep, eQuantType eq,
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
    p_slope = NULL ;


    set_version(net_param.version, &net_param) ;
	//Layers - Convolution
    if((e_conv == E_FC_FINAL) || (e_conv == E_FC))
    {
    	net_param.conv.window.h = 1 ;
    	net_param.conv.window.w = 1 ;
    	net_param.conv.same_pad = true ;
    }
    lconv = new BinFunc::Convolution(conv_depth, &(net_param.conv)) ;
   	//pooling
    if(e_pool == E_SUMPOOL){ lpool.s = new BinFunc::SumPooling(&(net_param.pool)) ; }
    else if(e_pool == E_MAXPOOL)
    {
	    assert(eq == E_ACTIVATION_SIGN) ; 
	    lpool.m = new BinFunc::MaxPooling(&(net_param.pool)) ; 
    }
   	//Quantizing
    if((e_activation == E_ACTIVATION_SIGN) || (net_param.quant.shift_bits < 1)){ net_param.quant.shift_bits = 1 ; }
    lquant = new BinFunc::Quantize(&(net_param.quant)) ; 

#ifdef _WEIGHT_CONVERT_
    //Batch Norm
    if(e_bias == E_BNORM){ bnorm = new BinFunc::BatchNorm(&(net_param.bnorm)) ; }
    else{ bnorm = NULL ; }
#endif
}

/*********************************************************************************
*    Func:      BinLayer::prep
*    Desc:      Prepares the binary layer, reading weights and allocating memory
*    Inputs:    FILE* - weights file 
*               tDimensions* - input dimensions
*    Return:    tDimensions* - output dimensions
*    Notes:     Input weights file is compressed for evaluation or TensorFlow for
*    		weight convert
*********************************************************************************/
tDimensions* BinLayer::prep(FILE* fd, tDimensions* dim)
{
    //allocate bias array
    p_bias = mbit_calloc(conv_depth, MULTIBIT_BITS, bk) ;
#ifndef ENCRYPTED
    p_slope = NULL ;
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM)){ p_slope = mbit_calloc(conv_depth, MULTIBIT_BITS, bk) ; }
#else
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM)){ p_slope = (uint32_t*) calloc(conv_depth, sizeof(uint32_t)) ; }
#endif
	//initialize arguments
    tActParams pact ;
    pact.d = dim ;
    //save/calculate dimensions and extract weights
    memcpy(&in_dim, pact.d, sizeof(in_dim)) ;
    pact.d = (tDimensions*) BinLayer::run(E_PREP, &pact, fd) ;
    memcpy(&out_dim, pact.d, sizeof(out_dim)) ;
#ifdef _WEIGHT_CONVERT_
    tActParams dummy ;
    void* d = BinLayer::run(E_PREP_BIAS, &dummy, fd) ;
#endif
    return pact.d ;
}

/*********************************************************************************
*    Func:      BinLayer::execute
*    Desc:      Runs all operations on the layer
*    Inputs:    tBit* - input to layer 
*    Return:    void* - layer outputs (either tBit* or tMultiBit*)
*    Notes:     None
*********************************************************************************/
void* BinLayer::execute(tBit* pb_in)
{
    tActParams pact ;
    pact.b = pb_in ;
    return  BinLayer::run(E_EXEC, &pact, NULL) ;
}

#ifdef _WEIGHT_CONVERT_
void BinLayer::export_weights(FILE* fd_bin)
{
    tActParams dummy ;
    BinLayer::run(E_EXPORT, &dummy, fd_bin) ;
}
#else
void BinLayer::export_weights(FILE* fd_bin){ printf("Weight convert not defined\r\n"); }
#endif

/*********************************************************************************
*    Func:      BinLayer::run
*    Desc:      Helper Function to prep, execute, export weights
*    Inputs:    eAction - either prepare or execute the laye
*    		tActParams - input parameters (either tDimiensions* or tBit*)
*    		FILE* - input weights file (weight convert prep), 
*    			output weights file (weight convert execute),
*    			or NULL (running neural network)
*    Return:    void* - layer outputs (either tDimensions*, tBit, or tMultiBit*)
*    Notes:     None
*********************************************************************************/
void* BinLayer::run(eAction e_act, tActParams* ap, FILE* fd)
{
    tBit* bdata  = (tBit*) ap->b ;
    tDimensions* dim = ap->d ;
    tMultiBit* mdata ;
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
            else if(e_act == E_EXEC){ mdata = lconv->execute(bdata) ; }
            //weight prep/export
            else if(e_act == E_PREP_BIAS){ lconv->extract_bias(fd, p_bias, e_bias) ; }
            else if(e_act == E_EXPORT){ lconv->export_weights(fd) ; }
        }
        break ;
        default: { assert(0) ;}
        break ;
    }
#ifdef _WEIGHT_CONVERT_
    if(e_bias == E_BNORM)
    {
        if(e_act == E_PREP){ dim = bnorm->prep(dim) ; }
        if(e_act == E_PREP_BIAS){ bnorm->extract_bias(fd, p_bias, p_slope) ; }
    }
#endif
    //Sumpooling
    if(e_pool == E_SUMPOOL)
    {
        if(e_act == E_PREP){ dim = lpool.s->prep(dim, bk) ; }
        else if(e_act == E_EXEC){ mdata = lpool.s->execute(mdata) ; }
        //weight prep
        else if(e_act == E_PREP_BIAS){ lpool.s->extract_bias(p_bias) ; }
    }

    //quantize
    if(e_act == E_PREP)
    {
        dim = lquant->prep(fd, dim, p_bias, p_slope, bk) ;
    }
    else if(e_act == E_EXEC)
    {
        if(e_activation == E_ACTIVATION_SIGN)
        {
            bdata = lquant->execute(mdata, p_bias) ;
        } else if(e_activation == E_ACTIVATION_NONE)
        {
            mdata = lquant->add_bias(mdata, p_bias) ;
	    } else if(e_activation == E_ACTIVATION_RELU)
	    {
	        mdata = (tMultiBit*) lquant->relu_shift(mdata, p_bias, p_slope) ;
	    }
    }
    //weight prep/export
    #ifndef ENCRYPTED
    else if(e_act == E_PREP_BIAS){ lquant->extract_bias(p_bias, p_slope) ; }
    else if(e_act == E_EXPORT){ lquant->export_weights(fd, p_bias, p_slope) ; }
    #endif
    
    if(e_activation == E_ACTIVATION_SIGN)
    {
    	//Max Pooling
        if(e_pool == E_MAXPOOL)
        {
            if(e_act == E_PREP){ dim = lpool.m->prep(dim, bk) ; }
            else if(e_act == E_EXEC){ bdata = lpool.m->execute(bdata) ; }
        }
    }

    //return
    if(e_act == E_PREP){ return (void*) dim ; }
    else if(e_act == E_EXEC)
    {
        if(e_activation != E_ACTIVATION_SIGN){ return (void*) mdata ; }
        else{ return (void*) bdata ; }
    }

    return (void*) NULL ;
}

/*********************************************************************************
*    Func:      BinLayer::set_version
*    Desc:      Modifies arguments for reverse compatibility
*    Inputs:    uint8_t - version
*    		tNetParams* - parameters to be modified
*    Return:    None
*    Notes:     None
*********************************************************************************/
//Version changes - reverse compatibility
void BinLayer::set_version(uint8_t v, tNetParams* net)
{
    if(v<1)
    {
        net->conv.stride.h = 1 ;
        net->conv.stride.w = 1 ;
        net->pool.stride.h = 0 ; 
        net->pool.stride.w = 0 ;
    }
}

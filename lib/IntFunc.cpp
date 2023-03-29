/***********************************************************************
* FILENAME :        IntFunc.cpp
*		
* VERSION:	    0.1
*
* DESCRIPTION :
*       This file contains all the Integer input layer functions for a
*  	a nerual network.
*
* PUBLIC CLASSES :
*       class IntFunc::Convolution Class
*	class IntFunc::BatchNorm
*	class IntFunc::SumPooling
*	class IntFunc::Quantize
*		
* NOTES :
*
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/

#include <cstdint>
#include <cstdio>
#include <assert.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <ratio>
#include <chrono>
#include <omp.h>
#include <iostream>
#include "IntFunc.h"
#include "Layer.h"
#include <cmath> //sqrt batch norm
#include <fstream>

#ifdef ENCRYPTED
#include "IntOps_enc.h"
#include "BinOps_enc.h"
#else
#include "IntOps.h"
#include "BinOps.h" //for get filters
#endif

#define SLOPE_BITS 8

using namespace std;
using namespace std::chrono;

/*********************************************************************************
*    Func:      IntFunc::Convolution::Convolution
*    Desc:      Integer Input Convolution Layer
*    Inputs:    uint32_t - layer output depth (# of neurons)
*               tConvParams* - convolution paramterts
*    Return:    Convolution Object
*    Notes:
*********************************************************************************/
IntFunc::Convolution::Convolution(uint16_t out_depth, tConvParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert(out_depth > 0) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    OutDepth = out_depth ;
    memcpy(&conv, in_params, sizeof(conv)) ;
    b_prep = false ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::prep
*    Desc:      Prepare for executing the convolution layer.
*               Allocates memory and preloads the weights
*    Inputs:    FILE* - weights (or convolution filter) file
*               tDimensions* - dimensions of input
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Convolution parameters (padding, window size, stride) are located
*               in the constructor
*********************************************************************************/
tDimensions* IntFunc::Convolution::prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk)
{
    assert(!b_prep) ;
    assert((ret_dim != NULL) && (fd_filt != NULL)) ;
    assert((ret_dim->hw.h >= conv.window.h ) && (ret_dim->hw.w >= conv.window.w)) ;
    assert((conv.stride.h != 0) && (conv.stride.w != 0)) ;

    //set parameters depth
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    bk = in_bk ;

    if(conv.same_pad)
    {   //zero padding -> offset is half of window size
        out_hw.h = (lay_dim.hw.h - 1)/conv.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/conv.stride.w + 1;
        if(conv.stride.h == 1){ offset_window.h = (int16_t)((conv.window.h-1)/2) ; }
	    else{ offset_window.h = (out_hw.h*conv.stride.h - lay_dim.hw.h)/2 ; }
        if(conv.stride.w == 1){ offset_window.w = (int16_t)((conv.window.w-1)/2) ; }
	    else{ offset_window.w = (out_hw.w*conv.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid padding - no offest, but remove left and right borders
    {
        offset_window.h = 0 ;
        offset_window.w = 0 ;
        out_hw.h = lay_dim.hw.h - 2*((int16_t)((conv.window.h-1)/2)) ;
        out_hw.h = (out_hw.h)/conv.stride.h  ;
        out_hw.w = lay_dim.hw.w - 2*((int16_t)((conv.window.w-1)/2)) ;
        out_hw.w = (out_hw.w)/conv.stride.w ;
    }

    //calculate number of additions for output bitsize
    in_up_bound = lay_dim.up_bound  ;
    lay_dim.up_bound *=
        (lay_dim.filter_bits) *
        (conv.window.w) *
        (conv.window.h) *
        (lay_dim.in_dep) ;

    for(lay_dim.out_bits = lay_dim.in_bits ;
        (lay_dim.up_bound >> lay_dim.out_bits) > 0 ;
        lay_dim.out_bits++) ;

    //allocate memory
    flen = get_size(&conv.window, lay_dim.in_dep, OutDepth) ;
    p_filters = (uint8_t*) calloc(flen, sizeof(uint8_t)) ;
    p_tern = (uint8_t*) calloc(flen, sizeof(uint8_t)) ;

    //get filters
    BinOps::get_ternfilters(fd_filt, p_filters, p_tern, flen, conv.tern_thresh, bk) ;

    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_dep = OutDepth ;
    ret_dim->in_bits = lay_dim.out_bits  ;
    ret_dim->out_bits = SINGLE_BIT ; //clear last dimension
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale ;

    b_prep = true ;
    return ret_dim ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::execute
*    Desc:      Performs Convolution
*    Inputs:    tFixedPoint* - image inputs to layer
*    Return:    tFixedPoint* - output of Convolution
*    Notes: 
*********************************************************************************/
tFixedPoint* IntFunc::Convolution::execute(tFixedPoint* p_inputs)
{
    tFixedPoint* p_inputs_bar ; //loop 1 result
    tFixedPoint* p_window ; //loop 2a processing
    tFixedPoint* p_output ; //loop 2b result

    //encrypted constant memory
    tFixedPoint* p_half_val ;
    tBit* p_zero_bit ;
#ifdef ENCRYPTED
    uint8_t* p_zero_bit_int = (uint8_t*)calloc(1, sizeof(uint8_t));
    *p_zero_bit_int = 0;
#endif
    p_half_val = fixpt_calloc(1, MULTIBIT_BITS, bk) ;
    p_zero_bit = bit_calloc(1, bk) ;
#ifdef ENCRYPTED
    for (int i = 0; i < MULTIBIT_BITS; i++) {
        lweClear(&p_half_val[0].ctxt[i], bk->bk->in_out_params);
    }
#else
    *p_half_val = 0 ;
#endif

    uint64_t input_i, filt_i, output_i ;
    uint8_t oob_i ;
    //shorter names for offset, convolution windows
    int16_t ofs_h = offset_window.h ;
    int16_t ofs_w = offset_window.w ;
    uint16_t st_h = conv.stride.h ;
    uint16_t st_w = conv.stride.w ;
    //number of items per output
    uint32_t partsum_ops = (conv.window.h)*(conv.window.w)*(lay_dim.in_dep) ;
    uint32_t partsum_ops_half = (partsum_ops+1)/2 ;

    //make sure model was prepared
    assert(b_prep) ;

#ifdef ENCRYPTED
    int dynamic_msg_space = BinOps::pow_int(2, lay_dim.out_bits);
    ofstream myfile;
    myfile.open("../../../client/bitsize.data");
    myfile << (int)lay_dim.out_bits << "\n";
    myfile.close();
#endif

    //allocate output memory
    p_inputs_bar = fixpt_calloc((lay_dim.hw.h)*(lay_dim.hw.w)*(lay_dim.in_dep), 1, bk);
    p_output = fixpt_calloc((lay_dim.hw.h)*(lay_dim.hw.w)*OutDepth, 1, bk) ;

    uint16_t bits, loop_cnt ; //bits
    uint64_t step;
    uint32_t od, ph, pw, fh, fw, di, wi;
    int curr_ctxt;
    bool oob;
    uint64_t ops, add_j;
    uint32_t threads = 8; // CONFIG: set to number of CPU threads desired

    //Loop 1 multiply, iterate over inputs
    #pragma omp parallel for collapse(3) private(input_i)
    for(ph = 0 ; ph < lay_dim.hw.h ; ph++)
    {
        for(pw = 0 ; pw < lay_dim.hw.w ; pw++)
        {
            //Loop 1a: multiply
            for(di = 0 ; di < lay_dim.in_dep ; di++)
            {
                input_i = get_input_i(ph, pw, di) ;
                #ifndef ENCRYPTED
                IntOps::invert(&p_inputs_bar[input_i], &p_inputs[input_i], p_zero_bit, bits, bk) ;
                #else
                BinOps::unbinarize_int(&p_inputs[input_i].ctxt[0], &p_inputs[input_i].ctxt[0], bk);
                lweClear(&p_inputs_bar[input_i].ctxt[0], bk->params->in_out_params);
                lweSubTo(&p_inputs_bar[input_i].ctxt[0], &p_inputs[input_i].ctxt[0], bk->params->in_out_params);
                #endif
            }
        }
    }

    //Loop 2: add
    #pragma omp parallel for collapse(3) private(di, fh, fw, p_window, oob_i, wi, oob, filt_i, input_i, bits, curr_ctxt, ops, step, add_j, loop_cnt, output_i)
    for(od = 0 ; od < OutDepth ; od++)
    {
        for(ph = 0 ; ph < out_hw.h ; ph++)
        {
            for(pw = 0 ; pw < out_hw.w ; pw++)
            {
                //allocate window
                p_window = fixpt_calloc(partsum_ops, 1, bk) ;
                oob_i = 0 ;
                //loop 3a: load
                for(wi = 0 ; wi < partsum_ops ; wi++)
                {
                    oob = retrieve_dims(wi, ph, pw, &di, &fh, &fw) ;
                    filt_i = get_filter_i(fh, fw, di, od) ;
                    input_i = get_input_i((fh+ph*st_h-ofs_h), (fw+pw*st_w-ofs_w), di) ;
                    bits  = lay_dim.in_bits ;
                    if(!oob && ((p_tern==NULL) || (p_tern[filt_i] == 0)))
                    {
                        if(p_filters[filt_i] == 0)
                        {
                            #ifdef ENCRYPTED
                            bootsCOPY(&(p_window[wi].ctxt[0]), &(p_inputs_bar[input_i].ctxt[0]), bk);
                            #else
                            p_window[wi] =  p_inputs_bar[input_i] ;
                            #endif
                        }
                        else
                        {
                            #ifdef ENCRYPTED
                            bootsCOPY(&(p_window[wi].ctxt[0]), &(p_inputs_bar[input_i].ctxt[0]), bk);
                            #else
                            p_window[wi] =  p_inputs[input_i] ;
                            #endif

                        }
                    }
                    //zero weight value, handled in bias
                    else if(p_tern != NULL && p_tern[filt_i] == 1)
                    {
                        #ifdef ENCRYPTED
                        lweClear(&(p_window[wi].ctxt[0]), bk->params->in_out_params);
                        #else
                        p_window[wi] =  *p_zero_bit ;
                        #endif
                    }
                    //padding, alternate 0s and 1s
		            else
                    {
                        #ifdef ENCRYPTED
                        lweClear(&(p_window[wi].ctxt[0]), bk->params->in_out_params);
                        #else
                        p_window[wi] =  *p_half_val ;
                        #endif
                    }
                }
                //loop 2b: add reduction
		        loop_cnt = 1 ;
                for(ops = partsum_ops ;  ops > 1 ; ops = (ops+1)/2)
                {
                    step = (1<<loop_cnt) ;
                    for(add_j = 0 ; add_j <  partsum_ops - step/2 ; add_j+=step)
                    {
#ifndef ENCRYPTED
                        IntOps::add(&p_window[add_j], &p_window[add_j], &p_window[add_j+step/2], bits, bk) ;
#else 
                        IntOps::add_inplace(&p_window[add_j], &p_window[add_j+step/2], MULTIBIT_BITS, bk) ;
#endif
                    }
                    loop_cnt++ ;
            		bits++;
                }
                output_i = get_output_i(ph, pw, od) ;
                #ifdef ENCRYPTED
                bootsCOPY(&p_output[output_i].ctxt[0], &p_window[0].ctxt[0], bk);
                #else
                p_output[output_i] = p_window[0] ;
                #endif
                fixpt_free(partsum_ops, p_window) ;
            }
        }
    }

    //free constants input data
    bit_free(1, p_zero_bit) ;
    fixpt_free(1, p_half_val) ;
    fixpt_free(lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep, p_inputs) ;
    fixpt_free((lay_dim.hw.h)*(lay_dim.hw.w)*(lay_dim.in_dep), p_inputs_bar);
    #ifdef ENCRYPTED
    free(p_zero_bit_int);
    #endif
    return p_output ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::retrieve_dims
*    Desc:      Maps picture pixels to convolutional windows
*    Inputs:    uint64_t - convolutional window pixel number
*               uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t* - returned input depth mapping
*               uint32_t* - returned convolution height loaction (y-value)
*               uint32_t* - returned convolution width loaction (x-value)
*    Return:    bool - true if input is "out of bounds" (oob)
*    Notes:   
*********************************************************************************/
bool inline IntFunc::Convolution::retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
                                        uint32_t* di, uint32_t* fh, uint32_t* fw)
{
    bool oob = false ;
    uint32_t cw_area = conv.window.h * conv.window.w ;
    *di = i/cw_area ;
    *fh = (i%cw_area)/conv.window.w  ;
    *fw = i%conv.window.w  ;
    //bounds check 1: make sure it is in bounds of num_ops
    if(i >= (lay_dim.in_dep*cw_area)){ oob = true ; }
    //bounds check 2: for same padding, make sure filter is in bounds
    if(conv.same_pad)
    {
        if(((uint32_t)((*fh)+ph*conv.stride.h-offset_window.h) >= lay_dim.hw.h) ||
            ((uint32_t)((*fw)+pw*conv.stride.w-offset_window.w) >= lay_dim.hw.w))
        {  oob = true ; }
    }
    return oob ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::get_input_i
*    Desc:      Gets flattened index of input image
*    Inputs:    uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened input array
*    Notes:
*********************************************************************************/
uint64_t inline IntFunc::Convolution::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::get_filter_i
*    Desc:      Gets flattened index of filter window
*    Inputs:    uint32_t - filter pixel height location (y-value)
*               uint32_t - filter pixel width location  (x-value)
*               uint32_t - input filter pixel depth location  (z-value)
*               uint32_t - output filter pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened filter (weights) array
*    Notes:   
*********************************************************************************/
uint64_t inline IntFunc::Convolution::get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od)
{
    return ((((fh)*conv.window.w + fw)*lay_dim.in_dep + di)*OutDepth + od) ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::get_output_i
*    Desc:      Gets flattened index of output picture
*    Inputs:    uint32_t - output pixel height location (y-value)
*               uint32_t - output pixel width location  (x-value)
*               uint32_t - output filter pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened output array
*    Notes:
*********************************************************************************/
uint64_t inline IntFunc::Convolution::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    return (((ph)*out_hw.w + pw)*OutDepth + od) ;
}


/*********************************************************************************
*    Func:      IntFunc::Convolution::extract_filter_bias
*    Desc:      Calculate bias produced by the ternary filters
*    Inputs:    tMultiBit* - pointer to the bias term
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
#ifdef _WEIGHT_CONVERT_
void IntFunc::Convolution::extract_filter_bias(tMultiBit* p_bias)
{
    //In convolution with weight {integer "-1"|bit 0}, we only invert (1's complement) to negate
    //Need to add 1 to complete 2's complement for multiplies
    for(uint32_t od = 0 ; od < OutDepth ; od++)
    {
        for(uint32_t di = 0 ; di < lay_dim.in_dep ; di++)
        {
            for(uint16_t fh = 0 ; (fh < conv.window.h) ; fh++)
            {
                for(uint16_t fw = 0 ; (fw < conv.window.w) ; fw++)
                {
                    uint32_t filt_i = get_filter_i(fh, fw, di, od) ;
                    if((p_tern == NULL) || (p_tern[filt_i] == 0))
                    {
                        //add 1 if filter is 0|"-1", ignore if it is 1|"1"
                        p_bias[od] += (1 - p_filters[filt_i]) ;
                    }
                } //end for fw
            } //end for fh
        } //end for di
    } //end for od
}


/*********************************************************************************
*    Func:      IntFunc::Convolution::extract_filter_bias
*    Desc:      Extracts the bias from the convolution layer
*    Inputs:    FILE* - tensorflow weight file
*               tMultiBit* - input/output bias filter
*               eBiasType - type of bias (None, BatchNorm, Additive)
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void IntFunc::Convolution::extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias)
{
    assert(b_prep) ;
    assert(p_bias != NULL) ;
    assert((e_bias != E_BIAS) || (fd_filt != NULL)) ;

    //invert+1
    extract_filter_bias(p_bias) ;

    //Read Bias
    if(e_bias == E_BIAS)
    {
        uint32_t out_len = OutDepth ;
        tMultiBit* read_bias = mbit_calloc(out_len, MULTIBIT_BITS, bk) ;
        BinOps::get_intfilters(fd_filt, read_bias, out_len, bk) ;
        mbit_free(out_len, read_bias) ;
    }
}
/*********************************************************************************
*    Func:      IntFunc::Convolution::export_weights
*    Desc:      Exports weights to file
*    Inputs:    FILE* - output weight file
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void IntFunc::Convolution::export_weights(FILE* fd)
{
    //export filters
    BinOps::export_tern(fd, p_filters, p_tern, flen, bk) ;
}

#else
//these functions should only be used by weight convert program
void IntFunc::Convolution::extract_filter_bias(tMultiBit* p_bias){ printf("Weight Convert not defined\r\n") ; }
void IntFunc::Convolution::extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias){ printf("Weight Convert not defined\r\n") ; }
void IntFunc::Convolution::export_weights(FILE* fd) { printf("Weight Convert not defined\r\n") ; }
#endif

#ifdef _WEIGHT_CONVERT_
/*********************************************************************************
*    Func:    IntFunc::BatchNorm::BatchNorm
*    Desc:    BatchNorm Layer
*    Inputs:  tBNormParams* - input parameters
*    Return:  BatchNorm Object
*    Notes:   BatchNorm is not used in inference. It is only used for calculating weights
*********************************************************************************/
IntFunc::BatchNorm::BatchNorm(tBNormParams* in_params)
{
    assert(in_params != NULL) ;
    assert(in_params->eps > 0) ;
    assert(!(in_params->use_scale)) ;
    memcpy(&bparams, in_params, sizeof(tBNormParams)) ;
}

/*********************************************************************************
*    Func:    IntFunc::BatchNorm::prep
*    Desc:    Prepare the BatchNorm Layer
*    Inputs:  tDimensions* - input dimensions
*    Return:  tDimensions* - output dimensions
*    Notes:   BatchNorm is not used in inference. It is only used for calculating weights
*             Dimensions are unchanged
*********************************************************************************/
tDimensions* IntFunc::BatchNorm::prep(tDimensions* ret_dim, uint8_t bit_scale)
{
    memcpy(&lay_dim, ret_dim, sizeof(tDimensions)) ;
    scale = (1<<(bit_scale-1))-1 ;
    return ret_dim ;
}

/*********************************************************************************
*    Func:    BinFunc::BatchNorm::extract_bias
*    Desc:    Reads BatchNorm parameters from tensorflow file
*             Converts parameters into BNN friendly weights
*    Inputs:  FILE* - tensorflow file
*             tMultiBit* - input/output bias
*             tMultiBit* - output slope pointer
*    Return:  None.
*    Notes:   BatchNorm is not used in inference. It is only used for calculating weights
*             Dealing with fixedpoint integers, we must scale to full percision
*********************************************************************************/
void IntFunc::BatchNorm::extract_bias(FILE* fd_filt, tMultiBit* p_bias, tMultiBit* p_slope)
{
    void* bk = NULL ;
    uint16_t flen = lay_dim.in_dep ;

    //allocate temporary arrays
    tMultiBit* p_gamma =  mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_beta =   mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_mean =   mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_stddev = mbit_calloc(flen, MULTIBIT_BITS, bk) ;

    //get gamma
    if(bparams.use_scale){ BinOps::get_intfilters(fd_filt, (tMultiBit*) p_gamma, flen, bk) ; }
    else { for(uint16_t i = 0 ; i < lay_dim.in_dep ; i++) { p_gamma[i] = 1.0 ; } }
        
    //get beta, mean
    BinOps::get_intfilters(fd_filt,(tMultiBit*) p_beta, flen, bk) ;
    BinOps::get_intfilters(fd_filt, (tMultiBit*)p_mean, flen, bk) ;

    //get stddev
    BinOps::get_intfilters(fd_filt, (tMultiBit*)p_stddev, flen, bk) ;  //variance
    for(uint16_t i = 0 ; i < lay_dim.in_dep ; i++) { p_stddev[i] = sqrt(p_stddev[i] + bparams.eps) ; }
    //find bias
    //(x-u)/(v+e)*g+b > N ; N = 0
    //=(x)-(u)+((b-N)*(v+e)/g) > 0
    for(uint16_t i = 0 ; i < lay_dim.in_dep ; i++)
    {
	    //compared to tensorflow, inputs are scaled
        p_bias[i] -= (lay_dim.scale*p_mean[i]) ;
        p_bias[i] += (lay_dim.scale*(p_beta[i])*(p_stddev[i])/(p_gamma[i])) ;
	    if(p_slope != NULL){ p_slope[i] = p_gamma[i]/p_stddev[i] ; }
    }

    mbit_free(flen, p_gamma) ;
    mbit_free(flen, p_beta) ;
    mbit_free(flen, p_mean) ;
    mbit_free(flen, p_stddev) ;
    return ;
}
#endif

/*********************************************************************************
*    Func:      IntFunc::SumPooling::SumPooling
*    Desc:      Generates SumPooling object
*    Inputs:    tPoolParams* - pooling parameters
*    Return:    SumPooling Object
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
IntFunc::SumPooling::SumPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }
}

/*********************************************************************************
*    Func:      IntFunc::SumPooling::prep
*    Desc:      Prepares sumpooling layer
*    Inputs:    tDimensions* - input dimensions
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
tDimensions* IntFunc::SumPooling::prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;
    bk = in_bk ;

    assert((pool.stride.h) != 0 && pool.stride.w != 0) ;

    //set output depth
    if(pool.same_pad) //round up - (partial pools used)
    {
        out_hw.h = (lay_dim.hw.h - 1)/(pool.stride.h) + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/(pool.stride.w) + 1 ;
 	    if(pool.stride.h == 1){ offset_window.h = (int16_t)((pool.window.h-1)/2) ; }
	    else{ offset_window.h = (out_hw.h*pool.stride.h - lay_dim.hw.h)/2 ; }
        if(pool.stride.w == 1){ offset_window.w = (int16_t)((pool.window.w-1)/2) ; }
	    else{ offset_window.w = (out_hw.w*pool.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid pad - truncate (partial pools ignored)
    {
	    offset_window.h = 0 ;
	    offset_window.w = 0 ;
        out_hw.h = (lay_dim.hw.h-((uint16_t)(pool.window.h/2))-1)/pool.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w-((uint16_t)(pool.window.w/2))-1)/pool.stride.w + 1 ;
    }

    //update upper bound and out bits
    lay_dim.up_bound *= (pool.window.w) * (pool.window.h) ;
    for(lay_dim.out_bits = lay_dim.in_bits ;
        (lay_dim.up_bound >> lay_dim.out_bits) > 0 ;
        lay_dim.out_bits++) ;

    //update return dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale*(pool.window.w)*(pool.window.h) ;
    
    b_prep = true ;

    return ret_dim ;
}

/*********************************************************************************
*    Func:      IntFunc::SumPooling::execute
*    Desc:      Performs SumPooling operation
*    Inputs:    tFixedPoint* - input image
*    Return:    tFixedPoint* - output image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
tFixedPoint* IntFunc::SumPooling::execute(tFixedPoint* p_inputs)
{
    //get temporary variables
    //input picture indexes
    tRectangle ip = {0,0} ; //input picture indexes
    uint64_t input_i, output_i ;
    uint16_t fh;
    uint16_t fw;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tFixedPoint* p_output = fixpt_calloc(len, FIXEDPOINT_BITS, bk) ;
    #ifdef ENCRYPTED
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < MULTIBIT_BITS; j++) {
            lweClear(&p_output[i].ctxt[j], bk->params->in_out_params);
        }
    }
    #endif
    #pragma omp parallel for collapse(3) shared(p_output, p_inputs) private(ip, output_i, fh, fw, input_i)
    for(uint32_t di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(uint16_t oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(uint16_t opw = 0 ; opw < out_hw.w; opw++)
            {
                //get input picture indexes
                ip.h = oph * (pool.stride.h) - offset_window.h ;
                ip.w = opw * (pool.stride.w) - offset_window.w ;
                output_i = get_output_i(oph, opw, di) ;
                for(fh = 0 ; (fh < (pool.window.h))
                    && ((ip.h+fh) < (lay_dim.hw.h)) ; fh++)
                {
		            if((ip.h + fh) < 0){ continue ; }
                    for(fw = 0 ; (fw < (pool.window.w))
                        && ((ip.w+fw) < lay_dim.hw.w) ; fw++)
                    {
		                if((ip.w + fw) < 0){ continue ; }
                        input_i = get_input_i(ip.h+fh, ip.w+fw, di) ;
                        #ifndef ENCRYPTED
                        IntOps::add(&p_output[output_i], &p_output[output_i], &p_inputs[input_i],
                                lay_dim.out_bits, bk) ;
                        #else
                        IntOps::add_inplace(&p_output[output_i], &p_inputs[input_i],
                                MULTIBIT_BITS, bk) ;
                        #endif
                    }
                }
            }
        }
    }

    fixpt_free((lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep), p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      IntFunc::SumPooling::get_input_i
*    Desc:      Gets flattened index of input image
*    Inputs:    uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened input image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
uint64_t inline IntFunc::SumPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}

/*********************************************************************************
*    Func:      IntFunc::SumPooling::SumPooling
*    Desc:      Gets flattened index of output image
*    Inputs:    uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened output image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
uint64_t inline IntFunc::SumPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //indepth == outdepth
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}

#ifdef _WEIGHT_CONVERT_
/*********************************************************************************
*    Func:      BinFunc::SumPooling::extract_bias
*    Desc:      Extracts the bias from the sumpooling layer
*    Inputs:    tMultiBit* - bias filter
*    Return:    None
*    Notes:     For weight preprocessing only
*               Corresponds to tensorflow AveragePooling operation. The bias
*               preforms this adjustment
*********************************************************************************/
void IntFunc::SumPooling::extract_bias(tMultiBit* p_bias)
{
    assert(p_bias != NULL) ;
    for(uint32_t i = 0 ; (i < lay_dim.in_dep) ; i++)
    {
        p_bias[i] *= (pool.window.h)*(pool.window.w) ;
    }
}
#else
//these functions should only be used by weight_convert program
void IntFunc::SumPooling::extract_bias(tMultiBit* p_bias){ printf("Weight convert not defined\r\n") ; }
#endif


/*********************************************************************************
*    Func:      IntFunc::Quantize::Quantize
*    Desc:      Binary Input Activation Layer
*    Inputs:    tQParams* - convolution paramterts
*    Return:    Quantize Object
*    Notes:     Contains both quantized relu and sign activations
**********************************************************************************/
IntFunc::Quantize::Quantize(tQParams* qparam)
{
    dim_len = SIZE_EMPTY ;
    shift_bits = qparam->shift_bits ;
    b_prep = false ;
}

/*********************************************************************************
*    Func:      IntFunc::Quantize::prep
*    Desc:      Prepare for executing the convolution layer.
*               Allocates memory and preloads the weights.
*    Inputs:    FILE* - input weights file
*               tDimensions* - input dimensions
*               tMultiBit* - input bias
*               tMultiBit* - slope (for ReLU)
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Contains both quantized relu and sign activations
**********************************************************************************/
#ifndef ENCRYPTED
tDimensions* IntFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim,
		tMultiBit* p_bias, tMultiBit* p_slope, TFheGateBootstrappingCloudKeySet* in_bk)
#else
tDimensions* IntFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim,
		tMultiBit* p_bias, uint32_t* p_slope, TFheGateBootstrappingCloudKeySet* in_bk)
#endif
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;

    //get bias offset
#ifdef _WEIGHT_CONVERT_
#else //!_WEIGHT_CONVERT_
    uint32_t bias_len = ret_dim->in_dep ;
    assert((fd_bias != NULL) && (p_bias != NULL)) ;
    BinOps::get_intfilters(fd_bias, (tMultiBit*) p_bias, bias_len, in_bk) ;
#ifndef ENCRYPTED
    if(p_slope!=NULL && shift_bits> 1){ BinOps::get_intfilters(fd_bias, (tMultiBit*) p_slope, bias_len, in_bk) ; }
#else
    if(p_slope!=NULL && shift_bits> 1){ BinOps::get_intfilters_ptxt(fd_bias, (uint32_t*) p_slope, bias_len) ; }
#endif

#endif //!_WEIGHT_CONVERT_

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    bk = in_bk ;

    //adjust scales
    uint8_t sc_b = 0 ; //scale_bits
    for(sc_b= 0 ; (1<<sc_b) < lay_dim.scale ; sc_b++); //log2(scale)
    slope_bits = SLOPE_BITS + sc_b - shift_bits ;      //adjust slope_bits (SLOPE_BITS is max slope value)


    //set output dimensions, other parameters
    if(shift_bits == 0)
    {   //no activation
        lay_dim.out_bits = lay_dim.in_bits ;
        ret_dim->up_bound = lay_dim.up_bound ;
        ret_dim->scale = lay_dim.scale ;
    } else if(shift_bits==1)
    {   //sign
        lay_dim.out_bits = 1 ;
        ret_dim->up_bound = 1 ;
#ifdef ZERO_BRIDGE
        ret_dim->scale = 0.5 ;
#else
	ret_dim->scale = 1.0 ;
#endif
    } else
    {   //relu shift
        lay_dim.out_bits = shift_bits ;
        ret_dim->up_bound = (1<<(lay_dim.out_bits)) - 1 ;
        ret_dim->scale = ret_dim->up_bound ; //scale is max value
    }
    dim_len = get_size(&lay_dim.hw, SIZE_EMPTY, lay_dim.in_dep) ;

    //update dimensions
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = (1<<(lay_dim.out_bits-1)) ;

    b_prep = true ;

    return ret_dim ;
}


/*********************************************************************************
*    Func:      IntFunc::Quantize::execute
*    Desc:      Sign Activation Layer
*    Inputs:    tFixedPoint* - input image (after convolution)
*               tMultiBit* - input bias
*    Return:    tBit* - output of sign activation
*    Notes:     Sign activation
**********************************************************************************/
tBit* IntFunc::Quantize::execute(tFixedPoint* p_inputs, tMultiBit* p_bias)
{
    tFixedPoint* x_add ;
    uint32_t di;
    tBit* p_output ;
    //input error checking
    assert(b_prep) ;
    //allocate memory
    p_output = bit_calloc(dim_len, bk) ;
    uint8_t flag = 0;

   #pragma omp parallel for shared(p_inputs, p_bias, p_output) private(di, x_add, flag)
    for(uint64_t i = 0 ; i < dim_len ; i++)
    {
        x_add = fixpt_calloc(1, 1 , bk) ;
        di = i % lay_dim.in_dep ;
#ifndef ENCRYPTED
        IntOps::add(&x_add[0], &p_inputs[i], (tFixedPoint*) &p_bias[di], lay_dim.in_bits+1, bk) ;
#else 
        IntOps::add(&x_add[0], &p_inputs[i], (tFixedPoint*) &p_bias[di], MULTIBIT_BITS, bk) ;
#endif
#ifndef ENCRYPTED
     	IntOps::binarize(&p_output[i], &x_add[0], lay_dim.in_bits+1, bk) ;
#else
     	BinOps::binarize_int(&p_output[i], &x_add[0].ctxt[0], 1, bk) ;
#endif
    }

    return p_output ;
}


/*********************************************************************************
*    Func:      IntFunc::Quantize::add_bias
*    Desc:      Add bias
*    Inputs:    tFixedPoint* - input image (after convolution)
*               tMultiBit* - input bias
*    Return:    tFixedPoint* - output image
*    Notes:     For output layers with no activation
**********************************************************************************/
tFixedPoint* IntFunc::Quantize::add_bias(tFixedPoint* p_inputs, tMultiBit* p_bias)
{
    tFixedPoint* p_output ;

    //input error checking
    assert(b_prep) ;
    //allocate memory
    p_output = fixpt_calloc(dim_len, MULTIBIT_BITS, bk) ;
    uint32_t di;
   #pragma omp parallel for shared(p_inputs, p_bias, p_output) private(di)
    for(uint32_t i = 0 ; i < dim_len ; i++)
    {
        di = i % lay_dim.in_dep ;
#ifndef ENCRYPTED
        IntOps::add(&p_output[i], &p_inputs[i], (tFixedPoint*) &p_bias[di], lay_dim.in_bits+1, bk) ;
#else
        IntOps::add(&p_output[i], &p_inputs[i], (tFixedPoint*) &p_bias[di], MULTIBIT_BITS, bk) ;
#endif
    }
    fixpt_free(dim_len, p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      IntFunc::Quantize::relu_shift
*    Desc:      Discretized ReLU
*    Inputs:    tFixedPoint* - input image (after convolution)
*               tMultiBit* - input bias
*               tMultiBit* - discretized slope of BatchNorm transform
*    Return:    tFixedPoint* - output image
*    Notes:     Uses DoReFa layer from Zhou et. al.
*               https://arxiv.org/abs/1606.06160
**********************************************************************************/
#ifndef ENCRYPTED
tFixedPoint* IntFunc::Quantize::relu_shift(tFixedPoint* p_inputs, tMultiBit* p_bias, tMultiBit* p_slope)
#else
tFixedPoint* IntFunc::Quantize::relu_shift(tFixedPoint* p_inputs, tMultiBit* p_bias, uint32_t* p_slope)
#endif
{
    tFixedPoint* x_bn ;
    tFixedPoint* p_output ;
    uint32_t di;
    //input error checking
    assert(b_prep) ;
    //allocate memory
    p_output = fixpt_calloc(dim_len, MULTIBIT_BITS, bk) ;
    int min = 00, max=100 ;
    #ifdef ENCRYPTED
    x_bn = fixpt_calloc(3, 1 , bk) ;
    lweClear(&x_bn[2].ctxt[0], bk->bk->in_out_params);
    #else
    x_bn = fixpt_calloc(1, FIXEDPOINT_BITS , bk) ;
    #endif
    #pragma omp parallel for shared(p_output, p_inputs) firstprivate(x_bn)
    for(uint64_t i = 0 ; i < dim_len ; i++)
    {
        di = i % lay_dim.in_dep ;
        #ifdef ENCRYPTED
        BinOps::multiply_pc_ints(&(x_bn[0].ctxt[0]), &(p_inputs[i].ctxt[0]), &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, bk) ;
        BinOps::add_int_inplace(&(x_bn[0].ctxt[0]),  &(p_bias[di].ctxt[0]), bk) ;
        bootsCOPY(&x_bn[1].ctxt[0], &(x_bn[0].ctxt[0]), bk);
        BinOps::binarize_int(&(x_bn[0].ctxt[0]), &(x_bn[0].ctxt[0]), lay_dim.in_bits, bk) ;
        bootsMUX(&(p_output[i].ctxt[0]), &(x_bn[0].ctxt[0]), &(x_bn[2].ctxt[0]), &(x_bn[1].ctxt[0]), bk);
        #else
        IntOps::multiply_pc_ints(&(x_bn[0]), &(p_inputs[i]), &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, bk) ;
        IntOps::add_pc_ints(&x_bn[0], &(x_bn[0]), &(p_bias[di]), (lay_dim.in_bits+SLOPE_BITS), bk) ;
        IntOps::shift(&(p_output[i]), &(x_bn[0]), (lay_dim.in_bits+SLOPE_BITS), slope_bits, bk) ;
        IntOps::relu(&(p_output[i]), &(p_output[i]), shift_bits, bk) ;
        #endif
    }

    fixpt_free(dim_len, p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      IntFunc::Quantize::extract_bias
*    Desc:      Extracts the bias from the activation layer
*    Inputs:    tMultiBit* - input bias
*               tMultiBit* - discretized slope of BatchNorm transform
*    Return:    None.
*    Notes:    
**********************************************************************************/
#ifdef _WEIGHT_CONVERT_
void IntFunc::Quantize::extract_bias(tMultiBit* p_bias, tMultiBit* p_slope)
{
    for(uint32_t od = 0 ; od < lay_dim.in_dep ; od++)
    {
        //Only if evaluating full batch norm
        if(p_slope!=NULL && lay_dim.out_bits> 1)
        {
            p_slope[od] *= (1<<(shift_bits))/lay_dim.scale ;  //adjust old scale with new scale
 		    p_slope[od] *= (1<<(slope_bits)) ; 		  //add percision
 		    p_bias[od] *= p_slope[od] ; 			  //adjust to slope 
 			//round amounts
 		    p_bias[od] += 0.5*(1<<(slope_bits)) + 0.5 ;       //add do-re-fa offset and round
 		    p_slope[od] += 0.5 ;   
	    }
    }
    return ;
}

/*********************************************************************************
*    Func:      IntFunc::Convolution::export_weights
*    Desc:      Exports weights to file
*    Inputs:    FILE* - output weight file
*               tMultiBit* - bias after convolution, batchnorm, pooling, pre-quantize
*               tMultiBit* - slope for ReLU; NULL if not ReLU activation
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void IntFunc::Quantize::export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope)
{
    //export filters
   uint32_t blen = lay_dim.in_dep ;
   BinOps::export_signedBias(fd, p_bias, blen, bk) ;
    //only run if we need slope - i.e. if we are ouputting more then one bit
    if(p_slope!=NULL && lay_dim.out_bits> 1)
    {
        BinOps::export_mulbits(fd, p_slope, blen, bk) ;
    }
}

#else
//these functions should only be used by weight_convert program
#ifndef ENCRYPTED
void IntFunc::Quantize::extract_bias(tMultiBit* p_bias, tMultiBit* p_slope){ printf("Weight convert not defined\r\n") ; }
void IntFunc::Quantize::export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope){ printf("Weight convert not defined\r\n") ; }
#endif
#endif

/***********************************************************************
* FILENAME :        BinFunc.cpp
*		
* VERSION:	    0.1
*
* DESCRIPTION :
*       This file contains all the binary input layer functions for a
*  	a nerual network.
*
* PUBLIC Classes :
*       class BinFunc::Convolution Class
*	class BinFunc::BatchNorm
*	class BinFunc::SumPooling
*	class BinFunc::MaxPooling
*	class BinFunc::Quantize
*		
* NOTES : 
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <chrono>
#include <iostream>
#include "Layer.h"
#include <omp.h>
#include <cmath>
#include "BinFunc.h"
#include <fstream>
#ifdef ENCRYPTED
#include "BinOps_enc.h"
#include "IntOps_enc.h"
#else
#include "IntOps.h"
#include "BinOps.h"
#endif
#define SLOPE_BITS 16
char layer_cnt_debug = 48; //FIXME
using namespace std;
using namespace std::chrono;

/*********************************************************************************
*    Func:    	BinFunc::Convolution
*    Desc:    	Binary Input Convolution Layer
*    Inputs:  	uint32_t - layer output depth (# of neurons)
*    		tConvParams* - convolution paramterts
*    Return:    Convolution Object
*    Notes:   
*********************************************************************************/
BinFunc::Convolution::Convolution(uint32_t out_dep, tConvParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert(out_dep > 0) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    OutDepth = out_dep ;
    memcpy(&conv, in_params, sizeof(conv)) ;
    b_prep = false ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::prep
*    Desc:    	Prepare for executing the convolution layer.
*    		Allocates memory and preloads the weights
*    Inputs:  	FILE* - weights (or convolution filter) file
*    		tDimensions* - dimensions of input
*    		TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Convolution parameters (padding, window size, stride) are located
*    		in the constructor
*********************************************************************************/
tDimensions* BinFunc::Convolution::prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk)
{
    assert(!b_prep) ;
    assert((ret_dim != NULL) && (fd_filt != NULL)) ;
    assert((conv.stride.h != 0) && (conv.stride.w != 0)) ;

    //set parameters depth
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    bk = in_bk ;

    if(conv.same_pad)
    {   //zero padding -offset is half of window size
        out_hw.h = (lay_dim.hw.h-1)/conv.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w-1)/conv.stride.w + 1 ;
        if(conv.stride.h == 1){ offset_window.h = (int16_t)((conv.window.h-1)/2) ; }
	    else { offset_window.h = (out_hw.h*conv.stride.h - lay_dim.hw.h)/2 ; }
        if(conv.stride.w == 1){ offset_window.w = (int16_t)((conv.window.w-1)/2) ; }
	    else{ offset_window.w = (out_hw.w*conv.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid padding - no offest, but remove left and right borders
    {
        offset_window.w = 0 ;
        offset_window.h = 0 ;
        out_hw.h = lay_dim.hw.h - 2*((int16_t)((conv.window.h-1)/2)) ;
        out_hw.h = out_hw.h/conv.stride.h ;
        out_hw.w = lay_dim.hw.w - 2*((int16_t)((conv.window.w-1)/2)) ;
        out_hw.w = out_hw.w/conv.stride.w ;
    }

    //calculate number of additions for output bitsize
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
    p_filters = new uint8_t [flen] ;
    p_tern = new uint8_t [flen] ;
    //get filters
    BinOps::get_ternfilters(fd_filt, p_filters, p_tern, flen, conv.tern_thresh, bk) ;
    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_dep = OutDepth ;
    ret_dim->in_bits = lay_dim.out_bits  ;
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale  ;
    ret_dim->out_bits = SINGLE_BIT ;

    b_prep = true ;
    return ret_dim ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::execute
*    Desc:    	Performs Convolution
*    Inputs:  	tBit* - image inputs to layer
*    Return:    tMultiBit* - output of Convolution
*    Notes: 
*********************************************************************************/
tMultiBit* BinFunc::Convolution::execute(tBit* p_inputs)
{
    tBit* p_inputs_bar ; //loop 1 result
    tMultiBit* p_window ; //loop 2a processing
    tMultiBit* p_output ; //loop 2b result

    //encrypted constant memory
#ifdef ENCRYPTED
    uint8_t p_zero_bit;
    uint8_t p_ones_bit;
    p_zero_bit = 0;
    p_ones_bit = 1;
#else
    tBit* p_zero_bit ;
    tBit* p_ones_bit ;
    p_zero_bit = bit_calloc(1, bk) ;
    p_ones_bit =  bit_calloc(1, bk) ;
    *p_ones_bit = 1 ;
#endif

    uint64_t input_i, filt_i, output_i ;
    uint8_t oob_i ;

#ifdef ENCRYPTED
    int dynamic_msg_space = BinOps::pow_int(2, lay_dim.out_bits);
    ofstream myfile;
    myfile.open("../../../client/bitsize.data");
    myfile << (int)lay_dim.out_bits << "\n";
    myfile.close();
#endif

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

    //allocate output memory
    p_inputs_bar = bit_calloc((lay_dim.hw.h)*(lay_dim.hw.w)*(lay_dim.in_dep), bk);
    p_output = mbit_calloc((lay_dim.hw.h)*(lay_dim.hw.w)*OutDepth, MULTIBIT_BITS, bk) ;

    uint16_t bits ; //bits
    uint64_t step;
    uint32_t od, ph, pw, fh, fw, di, wi;
    int curr_ctxt;
    bool oob;
    uint64_t ops, add_j;
    uint32_t threads = 8; // CONFIG: set to number of CPU threads desired

    //Loop 1 multiply, iterate over inputs
    #pragma omp parallel for collapse(3) shared(p_inputs, p_inputs_bar, p_zero_bit) private(input_i, di)
    for(ph = 0 ; ph < lay_dim.hw.h ; ph++)
    {
         for(pw = 0 ; pw < lay_dim.hw.w ; pw++)
         {
             //Loop 1a: multiply
             for(di = 0 ; di < lay_dim.in_dep ; di++)
             {
                input_i = get_input_i(ph, pw, di) ;
                //XNOR 0 is NOT
                //XNOR 1 is copy - use original input
                #ifdef ENCRYPTED
                BinOps::binarize_int(&p_inputs[input_i], &p_inputs[input_i], lay_dim.out_bits, bk);
                lweClear(&p_inputs_bar[input_i], bk->params->in_out_params);
                lweSubTo(&p_inputs_bar[input_i], &p_inputs[input_i], bk->params->in_out_params);
                #else
                BinOps::multiply(&p_inputs_bar[input_i], &p_inputs[input_i], p_zero_bit, bk) ;
                #endif
             }
         }
    }

    //Loop 2: add
    #pragma omp parallel for collapse(3) shared(p_output, p_filters, p_inputs, p_inputs_bar, p_tern, p_zero_bit, p_ones_bit) private(di, fh, fw, p_window, oob_i, wi, oob, filt_i, input_i, bits, curr_ctxt, ops, step, add_j, output_i)
    for(od = 0 ; od < OutDepth ; od++)
    {
        for(ph = 0 ; ph < out_hw.h ; ph ++)
        {
            for(pw = 0 ; pw < out_hw.w ; pw++)
            {
                //allocate window
                p_window = mbit_calloc(partsum_ops, 1, bk) ;
                oob_i = 0 ;
                //loop 3a: load
                for(wi = 0 ; wi < partsum_ops ; wi++)
                {
                    oob = retrieve_dims(wi, ph, pw, &di, &fh, &fw) ;
                    filt_i = get_filter_i(fh, fw, di, od) ;
                    input_i = get_input_i((fh+ph*st_h-ofs_h), (fw+pw*st_w-ofs_w), di) ;
                   
		    if(!oob && ((p_tern==NULL) || (p_tern[filt_i] == 0)))
                    {
                        if(p_filters[filt_i] == 0)
                        {
                            #ifdef ENCRYPTED
                            bootsCOPY(&(p_window[wi].ctxt[0]), &(p_inputs_bar[input_i]), bk);
                            #else
#ifdef ZERO_BRIDGE
                            p_window[wi] =  p_inputs_bar[input_i] ;
#else
                            p_window[wi] =  p_inputs_bar[input_i]==0?-1:1 ;
#endif
			    #endif
                        }
                        else
                        {
                            #ifdef ENCRYPTED
                            bootsCOPY(&(p_window[wi].ctxt[0]), &(p_inputs[input_i]), bk);
                            #else
#ifdef ZERO_BRIDGE
                            p_window[wi] =  p_inputs[input_i] ;
#else
                            p_window[wi] =  p_inputs[input_i]==0?-1:1 ;
#endif
                            #endif

                        }
                    }
                    else if(p_tern !=NULL && p_tern[filt_i] == 1)
                    {
                        #ifdef ENCRYPTED
                        lweClear(&(p_window[wi].ctxt[0]), bk->params->in_out_params);
                        #else
                        p_window[wi] =  *p_zero_bit ;
                        #endif
                    }
                    //padding, set to 0s
                    else
                    {
#ifdef ZERO_BRIDGE
			if(((oob_i++) % 2) == 0)
#else
                        if(true) //change - always set to 0
#endif
                        {
                            #ifdef ENCRYPTED
                            lweClear(&(p_window[wi].ctxt[0]), bk->params->in_out_params);
                            #else
                            p_window[wi] =  *p_zero_bit ;
                            #endif
                        }
                        else
                        {
                            #ifdef ENCRYPTED
                            const Torus32 mu = modSwitchToTorus32(1, dynamic_msg_space);
                            lweNoiselessTrivial(&(p_window[wi].ctxt[0]), mu, bk->params->in_out_params);
                            #else
                            p_window[wi] =  *p_ones_bit ;
                            #endif
                        }
                    }
	        }
            //loop 2b: add reduction
            bits  = 1 ;
            for(ops = partsum_ops ;  ops > 1 ; ops = (ops+1)/2)
            {
                step = (1<<bits) ;
                for(add_j = 0 ; add_j <  partsum_ops - step/2 ; add_j+=step)
                {
#ifdef ENCRYPTED
                BinOps::add_int_inplace(&p_window[add_j].ctxt[0], &p_window[add_j+step/2].ctxt[0], bk);
#else
                p_window[add_j] = p_window[add_j] + p_window[add_j + step/2] ;
#endif
		        }
                bits++;
            }
            output_i = get_output_i(ph, pw, od) ;
#ifdef ENCRYPTED
            bootsCOPY(&p_output[output_i].ctxt[0], &p_window[0].ctxt[0], bk);
#else
            p_output[output_i] = p_window[0] ;
#endif
            mbit_free(partsum_ops, p_window) ;
            }
        }
    }

    //free constants input data
    #ifndef ENCRYPTED
    bit_free(1, p_zero_bit) ;
    bit_free(1, p_ones_bit) ;
    #endif
    bit_free(lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep, p_inputs) ;
    bit_free((lay_dim.hw.h)*(lay_dim.hw.w)*(lay_dim.in_dep), p_inputs_bar);
    return p_output ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::retrieve_dims
*    Desc:    	Maps picture pixels to convolutional windows
*    Inputs:  	uint64_t - convolutional window pixel number
*  		uint32_t - pixel height location (y-value)
*  		uint32_t - pixel width location  (x-value)
*  		uint32_t* - returned input depth mapping
*  		uint32_t* - returned convolution height loaction (y-value)
*  		uint32_t* - returned convolution width loaction (x-value)
*    Return:    bool - true if input is "out of bounds" (oob)
*    Notes:   
*********************************************************************************/
bool inline BinFunc::Convolution::retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
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
*    Func:    	BinFunc::Convolution::get_input_i
*    Desc:    	Gets flattened index of input image
*    Inputs:    uint32_t - pixel height location (y-value)
*  		uint32_t - pixel width location  (x-value)
*  		uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened input array
*    Notes:   
*********************************************************************************/
uint64_t inline BinFunc::Convolution::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::get_filter_i
*    Desc:    	Gets flattened index of filter window
*    Inputs:    uint32_t - filter pixel height location (y-value)
*  		uint32_t - filter pixel width location  (x-value)
*  		uint32_t - input filter pixel depth location  (z-value)
*  		uint32_t - output filter pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened filter (weights) array
*    Notes:   
*********************************************************************************/
uint64_t inline BinFunc::Convolution::get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od)
{
    return ((((fh)*conv.window.w + fw)*lay_dim.in_dep + di)*OutDepth + od) ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::get_output_i
*    Desc:    	Gets flattened index of output picture
*    Inputs:    uint32_t - output pixel height location (y-value)
*  		uint32_t - output pixel width location  (x-value)
*  		uint32_t - output filter pixel depth location  (z-value)
*    Return:    uint64_t - index in flattened output array
*    Notes:   
*********************************************************************************/
uint64_t inline BinFunc::Convolution::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    return (((ph)*out_hw.w + pw)*OutDepth + od) ;
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::get_outhw
*    Desc:      Fetches output picture height width
*    Inputs:    tRectangle* - returned height and width of output
*    Return:    None
*    Notes:     For information/debugging purposes
*********************************************************************************/
void BinFunc::Convolution::get_outhw(tRectangle* ret_hw)
{
    memcpy(ret_hw, &out_hw, sizeof(out_hw));
}

/*********************************************************************************
*    Func:    	BinFunc::Convolution::get_outdep
*    Desc:      Fetches output picture depth
*    Inputs:    uint32_t* - returned out depth value
*    Return:    None
*    Notes:     For information/debugging purposes
*********************************************************************************/
void BinFunc::Convolution::get_outdep(uint32_t* out_dep)
{
    *out_dep = OutDepth ;
}


#ifdef _WEIGHT_CONVERT_
/*********************************************************************************
*    Func:    	BinFunc::Convolution::extract_filter_bias
*    Desc:    	Calculate bias produced by the ternary filters
*    Inputs:    tMultiBit* - pointer to the bias term
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void BinFunc::Convolution::extract_filter_bias(tMultiBit* p_bias)
{
    return ;
#ifdef ZERO_BRIDGE
    //Need to add 1/2 for all of the ternary values (i.e. ternary is 1)
    for(uint64_t od = 0 ; od < OutDepth ; od++)
    {
        //add one for every MAC...
        for(uint64_t di = 0 ; di < lay_dim.in_dep ; di++)
        {
            for(uint64_t fh = 0 ; (fh < conv.window.h) ; fh++)
            {
                for(uint64_t fw = 0 ; (fw < conv.window.w) ; fw++)
                {
                    uint64_t filt_i = get_filter_i(fh, fw, di, od) ;
                    //add 1/2 for every integer "0" filter
                    p_bias[od] += (p_tern[filt_i]/2.0) ;
                }
            }
        }
    }
#endif
}

/*********************************************************************************
*    Func:      BinFunc::Convolution::extract_filter_bias
*    Desc:      Extracts the bias from the convolution layer
*    Inputs:    FILE* - tensorflow weight file
*               tMultiBit* - input/output bias filter
*               eBiasType - type of bias (None, BatchNorm, Additive)
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void BinFunc::Convolution::extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias)
{
    assert(p_bias != NULL) ;
    assert((e_bias != E_BIAS) || (fd_filt != NULL));
    //extract ternary weights
    extract_filter_bias(p_bias) ;
    //bias terms
    if(e_bias == E_BIAS)
    {
        //Read Bias
        uint64_t out_len = OutDepth ;
        tMultiBit* read_bias = mbit_calloc(out_len, MULTIBIT_BITS, bk) ;
        BinOps::get_intfilters(fd_filt, read_bias, out_len, bk) ;
        for(uint64_t i = 0 ; i < out_len; i++)
        {
            p_bias[i] += read_bias[i] ;
        }
        mbit_free(out_len, read_bias) ;
    }
}

/*********************************************************************************
*    Func:      BinFunc::Convolution::export_weights
*    Desc:      Exports weights to file
*    Inputs:    FILE* - output weight file
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void BinFunc::Convolution::export_weights(FILE* fd)
{
    //export filters
    BinOps::export_tern(fd, p_filters, p_tern, flen, bk) ;
}
#else
//these functions should only be used by weight_convert program
void BinFunc::Convolution::extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias){ printf("Weight convert not defined\r\n") ; }
void BinFunc::Convolution::export_weights(FILE* fd){ printf("Weight convert not defined\r\n") ; }
#endif

#ifdef _WEIGHT_CONVERT_
/*********************************************************************************
*    Func:    BinFunc::BatchNorm::BatchNorm
*    Desc:    BatchNorm Layer
*    Inputs:  tBNormParams* - input parameters
*    Return:  BatchNorm Object
*    Notes:   BatchNorm is not used in inference. It is only used for calculating weights
*********************************************************************************/
BinFunc::BatchNorm::BatchNorm(tBNormParams* in_params)
{
    assert(in_params != NULL) ;
    assert(in_params->eps > 0) ;
    assert(!(in_params->use_scale)) ;
    memcpy(&bparams, in_params, sizeof(tBNormParams)) ;
}

/*********************************************************************************
*    Func:    BinFunc::BatchNorm::prep
*    Desc:    Prepare the BatchNorm Layer
*    Inputs:  tDimensions* - input dimensions
*    Return:  tDimensions* - output dimensions
*    Notes:   BatchNorm is not used in inference. It is only used for calculating weights
*             Dimensions are unchanged
*********************************************************************************/
tDimensions* BinFunc::BatchNorm::prep(tDimensions* ret_dim)
{
    memcpy(&lay_dim, ret_dim, sizeof(tDimensions)) ;
    lay_dim.out_bits = lay_dim.in_bits ;
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
*********************************************************************************/
void BinFunc::BatchNorm::extract_bias(FILE* fd_filt, tMultiBit* p_bias, tMultiBit* p_slope)
{
    uint64_t flen = lay_dim.in_dep ;
    void* bk = NULL ;
    //allocate temporary arrays
    tMultiBit* p_gamma =  mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_beta =   mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_mean =   mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    tMultiBit* p_stddev = mbit_calloc(flen, MULTIBIT_BITS, bk) ;
    //get gamma
    if(bparams.use_scale){ BinOps::get_intfilters(fd_filt, p_gamma, flen, bk) ; }
    else { for(uint64_t i = 0 ; i < lay_dim.in_dep ; i++) { p_gamma[i] = 1.0 ; } }
    //get beta, mean
    BinOps::get_intfilters(fd_filt, p_beta, flen, bk) ;
    BinOps::get_intfilters(fd_filt, p_mean, flen, bk) ;
    //get stddev
    BinOps::get_intfilters(fd_filt, p_stddev, flen, bk) ;  //variance
    for(uint64_t i = 0 ; i < flen ; i++) { p_stddev[i] = sqrt(p_stddev[i] + bparams.eps) ; }

    //find bias
    //  (x-u)/(v+e)*g+b > N ; N = 0
    //= (x)-(u)+((b-N)*(v+e)/g) > 0
    for(uint64_t i = 0 ; i < flen ; i++)
    {
// Comments for ZERO_BRIDGE
        //need to divide by two to convert between unsigned bit and integer formats
        //ub = 2*si - dim->up_bound ; si = ub/2 + dim->up_bound/2
	//lay_dim.scale should equal 0.5
//-1 bridge scale should be 1
	p_bias[i] -= lay_dim.scale*p_mean[i] ;
        p_bias[i] += (lay_dim.scale*(p_beta[i])*(p_stddev[i])/(p_gamma[i])) ;
	if(p_slope!=NULL){ p_slope[i] = p_gamma[i]/p_stddev[i] ; }
    }

    mbit_free(flen, p_gamma) ;
    mbit_free(flen, p_beta) ;
    mbit_free(flen, p_mean) ;
    mbit_free(flen, p_stddev) ;

    return ;
}
#endif
/*********************************************************************************
*    Func:      BinFunc::SumPooling::SumPooling
*    Desc:      Generates SumPooling object
*    Inputs:    tPoolParams* - pooling parameters
*    Return:    SumPooling Object
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
BinFunc::SumPooling::SumPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }

    b_prep = false ;
}

/*********************************************************************************
*    Func:      BinFunc::SumPooling::prep
*    Desc:      Prepares sumpooling layer
*    Inputs:    tDimensions* - input dimensions
* 		TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
tDimensions* BinFunc::SumPooling::prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;
    assert((pool.stride.h) != 0 && pool.stride.w != 0) ;

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    bk = in_bk ;

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
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale * (pool.window.w) * (pool.window.h) ;
    ret_dim->out_bits = SINGLE_BIT ;

    b_prep = true ;
    return ret_dim ;
}
/*********************************************************************************
*    Func:      BinFunc::SumPooling::execute
*    Desc:      Performs SumPooling operation
*    Inputs:    tMultiBit* - input image
*    Return:    tMultiBit* - output image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
tMultiBit* BinFunc::SumPooling::execute(tMultiBit* p_inputs)
{
    //get temporary variables
    //input picture indexes
    tRectangle ip = {0,0} ; //input picture indexes
    uint64_t input_i, output_i ;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tMultiBit* p_output = mbit_calloc(len, 1, bk) ;
    #ifdef ENCRYPTED
    for (int i = 0; i < len; i++) {
        lweClear(&p_output[i].ctxt[0], bk->params->in_out_params);
    }
    #endif
    uint32_t fh;
    uint32_t fw;

    #pragma omp parallel for collapse(3) shared(p_output, p_inputs) private(ip, output_i, fh, fw, input_i)
    for(uint32_t di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(uint32_t oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(uint32_t opw = 0 ; opw < out_hw.w; opw++)
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
                        #ifdef ENCRYPTED
                        BinOps::add_int_inplace(&p_output[output_i].ctxt[0], &p_inputs[input_i].ctxt[0], bk);
                        #else
                        BinOps::add(&p_output[output_i], &p_output[output_i], &p_inputs[input_i],
                                lay_dim.out_bits, bk) ;
                        #endif
                    }
                }
            }
        }
    }

    mbit_free((lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep), p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      BinFunc::SumPooling::get_input_i
*    Desc:    	Gets flattened index of input image
*    Inputs:	uint32_t - pixel height location (y-value)
*  		uint32_t - pixel width location  (x-value)
*  		uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened input image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
uint64_t inline BinFunc::SumPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}

/*********************************************************************************
*    Func:      BinFunc::SumPooling::SumPooling
*    Desc:    	Gets flattened index of output image
*    Inputs:	uint32_t - pixel height location (y-value)
*  		uint32_t - pixel width location  (x-value)
*  		uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened output image
*    Notes:     Corresponds to tensorflow AveragePooling operation
*********************************************************************************/
uint64_t inline BinFunc::SumPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}

/*********************************************************************************
*    Func:      BinFunc::SumPooling::get_outhw
*    Desc:      Fetches output picture height width
*    Inputs:    tRectangle* - return pointer of picture dimensions
*    Return:    None
*    Notes:     For information/debugging purposes
*********************************************************************************/
void BinFunc::SumPooling::get_outhw(tRectangle* ret_hw)
{
    memcpy(ret_hw, &out_hw, sizeof(out_hw));
}
/*********************************************************************************
*    Func:      BinFunc::SumPooling::SumPooling
*    Desc:      Fetches output picture depth
*    Inputs:    uint32_t* - returned pointer of output depth
*    Return:    None
*    Notes:     For information/debugging purposes
*********************************************************************************/
void BinFunc::SumPooling::get_outdep(uint32_t* out_dep)
{
    *out_dep = lay_dim.in_dep ;
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
void BinFunc::SumPooling::extract_bias(tMultiBit* p_bias)
{
    assert(p_bias != NULL) ;
    for(uint32_t i = 0 ; (i < lay_dim.in_dep) ; i++)
    {
        p_bias[i] *= (pool.window.h)*(pool.window.w) ;
    }
}
#else
//these functions should only be used by weight_convert program
void BinFunc::SumPooling::extract_bias(tMultiBit* p_bias){ printf("Weight convert not defined\r\n") ; }
#endif

/*********************************************************************************
*    Func:      BinFunc::MaxPooling::MaxPooling
*    Desc:      Generates MaxPooling object
*    Inputs:    tPoolParams* - pooling parameters
*    Return:    MaxPooling Object
*    Notes:     Only possible if using the sign activation function
*********************************************************************************/
BinFunc::MaxPooling::MaxPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }

    b_prep = false ;
}

/*********************************************************************************
*    Func:      BinFunc::MaxPooling::prep
*    Desc:      Prepares maxpooling layer
*    Inputs:    tDimensions* - input dimensions
*               TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:    tDimensions* - output dimensions
*    Notes:     Only possible if using the sign activation function
*********************************************************************************/
tDimensions* BinFunc::MaxPooling::prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;
    assert((pool.stride.h) != 0 && pool.stride.w != 0) ;
    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    bk = in_bk ;

    //set output dimensions
    if(pool.same_pad) //round up - (partial pools used)
    {
        out_hw.h = (lay_dim.hw.h - 1)/(pool.stride.h) + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/(pool.stride.w) + 1 ;
    }
    else //valid pad - truncate (partial pools ignored)
    {
        out_hw.h = (lay_dim.hw.h)/(pool.window.h) ; 
        out_hw.w = (lay_dim.hw.w)/(pool.window.w) ;
    }

    //max value stays the same
    lay_dim.out_bits = lay_dim.in_bits ;

    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->scale = lay_dim.scale ;
    b_prep = true ;

    return ret_dim ;
}
/*********************************************************************************
*    Func:      BinFunc::MaxPooling::execute
*    Desc:      Performs MaxPooling operation
*    Inputs:    tMultibit* - input image
*    Return:    tMultiBit* - output image
*    Notes:     Only possible if using the sign activation function
*********************************************************************************/
tBit* BinFunc::MaxPooling::execute(tBit* p_inputs)
{
    //input picture indexes
    tRectangle ip = {0,0} ;
    uint64_t input_i, output_i ;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tBit* p_output = bit_calloc(len, bk) ;
    uint32_t fh;
    uint32_t fw;

    uint32_t ctr = 0;
    #pragma omp parallel for collapse(3) shared(p_output, p_inputs) private(ip, output_i, fh, fw, input_i)
    for(uint32_t di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(uint32_t oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(uint32_t opw = 0 ; opw < out_hw.w; opw++)
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
                        BinOps::max(&p_output[output_i], &p_output[output_i], &p_inputs[input_i], bk) ;
		            }
                }
            }
        }
    }
    bit_free(lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep, p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      BinFunc::MaxPooling::get_input_i
*    Desc:      Gets flattened index of input image
*    Inputs:    uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened input image
*    Notes:     Only possible if using the sign activation function
*********************************************************************************/
uint64_t inline BinFunc::MaxPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}

/*********************************************************************************
*    Func:      BinFunc::MaxPooling::get_output_i
*    Desc:      Gets flattened index of output image
*    Inputs:    uint32_t - pixel height location (y-value)
*               uint32_t - pixel width location  (x-value)
*               uint32_t - pixel depth location  (z-value)
*    Return:    uint64_t - index of flattened output image
*    Notes:     Only possible if using the sign activation function
*********************************************************************************/
uint64_t inline BinFunc::MaxPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //indepth == outdepth
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}


/*********************************************************************************
*    Func:      BinFunc::Quantize::Quantize
*    Desc:      Binary Input Activation Layer
*    Inputs:    tQParams* - convolution paramterts
*    Return:    Quantize Object
*    Notes:     Contains both quantized relu and sign activations
**********************************************************************************/
BinFunc::Quantize::Quantize(tQParams* qparam)
{
    assert(qparam!=NULL && qparam->shift_bits > 0);
    dim_len = SIZE_EMPTY ;
    shift_bits = qparam->shift_bits ;
    b_prep = false ;
}

/*********************************************************************************
*    Func:      BinFunc::Quantize::prep
*    Desc:      Prepare for executing the convolution layer.
*               Allocates memory and preloads the weights.
*    Inputs:    FILE* - input weights file
*               tDimensions* - input dimensions
*               tMultiBit* - input bias
*               tMultiBit* - slope (for ReLU)
*		TFheGateBootstrappingCloudKeySet* - bootstrapping key
*    Return:	tDimensions* - output dimensions
*    Notes:     Contains both quantized relu and sign activations
**********************************************************************************/
#ifdef ENCRYPTED
tDimensions* BinFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, uint32_t* p_slope, TFheGateBootstrappingCloudKeySet* in_bk)
#else
tDimensions* BinFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, tMultiBit* p_slope, TFheGateBootstrappingCloudKeySet* in_bk)
#endif
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;

    //get bias offset
#ifdef _WEIGHT_CONVERT_
#else
    uint32_t bias_len = ret_dim->in_dep ;
    assert((fd_bias != NULL) && (p_bias != NULL)) ;
    BinOps::get_intfilters(fd_bias, p_bias, bias_len, in_bk) ;
    if(p_slope != NULL)
    {
        #ifdef ENCRYPTED
	    BinOps::get_intfilters_ptxt(fd_bias, p_slope, bias_len) ;
        #else
	    BinOps::get_intfilters(fd_bias, p_slope, bias_len, in_bk) ;
        #endif
    }
#endif
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;
    //copy dimension
    bk = in_bk ;

    uint8_t sb = 0 ;
    for(sb= 0 ; (1<<sb) < sqrt(lay_dim.up_bound)/2 ; sb++);
    slope_bits = SLOPE_BITS + sb ;


    //set output dimensions, other parameters
    lay_dim.out_bits = (shift_bits>1)?(shift_bits+1):1 ;
    dim_len = get_size(&lay_dim.hw, SIZE_EMPTY, lay_dim.in_dep) ;

    //update dimensions
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = (1<<(lay_dim.out_bits-1)) ;
#ifdef ZERO_BRIDGE
    ret_dim->scale = (shift_bits>1) ? (ret_dim->up_bound) : 0.5  ;
#else
    ret_dim->scale = (shift_bits>1) ? (ret_dim->up_bound) : 0.5  ;
#endif
    b_prep = true ;

    return ret_dim ;
}

/*********************************************************************************
*    Func:      BinFunc::Quantize::execute
*    Desc:      Sign Activation Layer
*    Inputs:    tMultiBit* - input image (after convolution)
*               tMultiBit* - input bias
*    Return:    tBit* - output of sign activation
*    Notes:     Sign activation
**********************************************************************************/
tBit* BinFunc::Quantize::execute(tMultiBit* p_inputs, tMultiBit* p_bias)
{
    tBit* p_output ;
    tMultiBit* x_add ;
    uint8_t flag = 0;

    //input error checking
    assert(b_prep) ;
    //allocate memory
    p_output = bit_calloc(dim_len, bk) ;
    uint32_t di;

    #pragma omp parallel for shared(p_inputs, p_bias, p_output) private(flag, di, x_add)
    for(uint32_t i = 0 ; i < dim_len ; i++)
    {
        if (!flag) {
            x_add = mbit_calloc(1, 1, bk) ;
            flag = 1;
        }
        di = i % lay_dim.in_dep ;
        #ifdef ENCRYPTED
        BinOps::add_int(&x_add[0].ctxt[0], &(p_inputs[i].ctxt[0]), &p_bias[di].ctxt[0], bk) ;
        BinOps::binarize_int(&(p_output[i]), &x_add[0].ctxt[0], 1, bk);
        #else
	    BinOps::add(&x_add[0], &(p_inputs[i]), &p_bias[di], (lay_dim.in_bits), bk) ;
        BinOps::binarize(&(p_output[i]), &x_add[0], (lay_dim.in_bits), bk) ;
        #endif
    }

    mbit_free(dim_len, p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      BinFunc::Quantize::add_bias
*    Desc:      Add bias
*    Inputs:    tMultiBit* - input image (after convolution)
*               tMultiBit* - input bias
*    Return:    tMultiBit* - output image
*    Notes:     For output layers with no activation
**********************************************************************************/
tMultiBit* BinFunc::Quantize::add_bias(tMultiBit* p_inputs, tMultiBit* p_bias)
{
    tMultiBit* p_output ;

    //input error checking
    assert(b_prep) ;
    //allocate memory
    p_output = mbit_calloc(dim_len, MULTIBIT_BITS, bk) ;
    uint32_t di;

    #pragma omp parallel for shared(p_inputs, p_bias, p_output) private(di)
    for(uint32_t i = 0 ; i < dim_len ; i++)
    {
        di = i % lay_dim.in_dep ;
        #ifdef ENCRYPTED
        BinOps::add_int(&p_output[i].ctxt[0], &(p_inputs[i].ctxt[0]), &p_bias[di].ctxt[0], bk) ;
        #else
        BinOps::add(&p_output[i], &(p_inputs[i]), &p_bias[di], (lay_dim.in_bits+2), bk) ;
        #endif
    }
    mbit_free(dim_len, p_inputs) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      BinFunc::Quantize::relu_shift
*    Desc:      Discretized ReLU
*    Inputs:    tMultiBit* - input image (after convolution)
*               tMultiBit* - input bias
*               tMultiBit* - discretized slope of BatchNorm transform
*    Return:    tMultiBit* - output image
*    Notes:     Uses DoReFa layer from Zhou et. al.
*    		https://arxiv.org/abs/1606.06160
**********************************************************************************/
#ifdef ENCRYPTED
tFixedPoint* BinFunc::Quantize::relu_shift(tMultiBit* p_inputs, tMultiBit* p_bias, uint32_t* p_slope)
#else
tFixedPoint* BinFunc::Quantize::relu_shift(tMultiBit* p_inputs, tMultiBit* p_bias, tMultiBit* p_slope)
#endif
{
    tFixedPoint* p_output ;
    tMultiBit* x_bn ;
    tFixedPoint* x_fp ;

    assert(b_prep) ;
    //allocate memory
    p_output = fixpt_calloc(dim_len, MULTIBIT_BITS, bk) ;
    uint32_t di;
    uint8_t flag = 0;

    #pragma omp parallel for shared(p_inputs, p_bias, p_output) private(flag, di, x_fp, x_bn)
    for(uint32_t i = 0 ; i < dim_len ; i++)
    {
        if (!flag) {
          x_bn = mbit_calloc(1, MULTIBIT_BITS, bk) ;
          x_fp = fixpt_calloc(1, MULTIBIT_BITS, bk) ;
          flag = 1;
        }
        di = i % lay_dim.in_dep ;
        #ifdef ENCRYPTED
      	BinOps::multiply_pc_ints(&(x_bn[0].ctxt[0]), &(p_inputs[i].ctxt[0]), &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, bk) ;
      	BinOps::add_int(&x_fp[0].ctxt[0], &(x_bn[0].ctxt[0]),  &(p_bias[di].ctxt[0]), bk) ;
        BinOps::binarize_int(&(x_fp[0].ctxt[0]), &(x_fp[0].ctxt[0]), 1, bk) ;
        IntOps::shift(&(x_fp[0]), &(x_fp[0]), slope_bits, shift_bits+1, bk) ;
        IntOps::relu(&(p_output[i]), &(x_fp[0]), shift_bits, bk) ;
        BinOps::unbinarize_int(&(p_output[i].ctxt[0]), &(p_output[i].ctxt[0]), bk) ;
        #else
      	BinOps::multiply_pc_ints(&(x_bn[0]), &(p_inputs[i]), &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, bk) ;
      	BinOps::add_pc_ints(&x_fp[0], &(x_bn[0]),  &(p_bias[di]), (lay_dim.in_bits+SLOPE_BITS), bk) ;
        IntOps::shift(&(x_fp[0]), &(x_fp[0]), slope_bits, shift_bits+1, bk) ;
        IntOps::relu(&(p_output[i]), &(x_fp[0]), shift_bits, bk) ;
        #endif
    }

    mbit_free(dim_len, p_inputs) ;
    mbit_free(1, x_bn) ;
    return p_output ;
}

/*********************************************************************************
*    Func:      BinFunc::Quantize::extract_bias
*    Desc:      Extracts the bias from the activation layer
*    Inputs:    tMultiBit* - input bias
*               tMultiBit* - discretized slope of BatchNorm transform
*    Return:    None.
*    Notes:    
**********************************************************************************/
#ifdef _WEIGHT_CONVERT_
void BinFunc::Quantize::extract_bias(tMultiBit* p_bias, tMultiBit* p_slope)
{
#ifdef ZERO_BRIDGE
    float add_offset = uint32_t(1<<(lay_dim.in_bits-1)) - float(lay_dim.up_bound/2.0) ;
#else
    float add_offset = 0 ; //can remove
#endif
    for(uint32_t od = 0 ; od < lay_dim.in_dep ; od++)
    {
 	    //Only if evaluating full batch norm
	if(p_slope!=NULL)
	{
		    //need to convert from binary to integer domains
		    //normally, i = 2b-1
		    //Here, i = 2mx - m*(u/2) + mb where u is up_bound, m is slope, b is bias

		    //first lets scale the slope to prevent rounding errors
		    //slope is inversly proportional to variance, determined by depth of circuit
		    p_slope[od] *= (1<<slope_bits) ; //add some extra percision

		    //now lets find mb
		    p_bias[od] += 1.0/(1<<shift_bits) ;  //adjust to dorefa net
		    p_bias[od] *= p_slope[od] ; //adjust to exact number, adjust add_offset
		    //and m(u/2), plus an offset to keep our value positive
		    add_offset = -float(lay_dim.up_bound*p_slope[od]/2.0) ;
		    p_slope[od] +=0.5 ; //round slope for multiplcation with x
        }
	p_bias[od] += add_offset ;
   }
}

/*********************************************************************************
*    Func:      BinFunc::Convolution::export_weights
*    Desc:      Exports weights to file
*    Inputs:    FILE* - output weight file
*    		tMultiBit* - bias after convolution, batchnorm, pooling, pre-quantize
*    		tMultiBit* - slope for ReLU; NULL if not ReLU activation
*    Return:    None
*    Notes:     For weight preprocessing only
*********************************************************************************/
void BinFunc::Quantize::export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope)
{
    //export filters
    uint32_t blen = lay_dim.in_dep ;
    BinOps::export_mulbits(fd, p_bias, blen, bk) ;
    //only run if we need slope - i.e. if we are ouputting more then one bit
    if(p_slope!=NULL)
    {
	    BinOps::export_mulbits(fd, p_slope, blen, bk) ;
    }
}

#else
//these functions should only be used by weight_convert program
#ifndef ENCRYPTED
void BinFunc::Quantize::extract_bias(tMultiBit* p_bias, tMultiBit* p_slope){ printf("Weight convert not defined\r\n") ; }
void BinFunc::Quantize::export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope){ printf("Weight convert not defined\r\n") ; }
#endif
#endif
#ifdef ZERO_BRIDGE
#warning "Zero Bridge is defined"
#endif

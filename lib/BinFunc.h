/***********************************************************************
* FILENAME :        BinFunc.h
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file contains all the binary input layer functions for a
*       a nerual network.
*
* PUBLIC Classes :
*       class BinFunc::Convolution Class
*       class BinFunc::BatchNorm
*       class BinFunc::SumPooling
*       class BinFunc::MaxPooling
*       class BinFunc::Quantize
*               
* NOTES : 
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/

#ifndef _BIN_FUNC_H_
#define _BIN_FUNC_H_

#include <cstdio>
#include "Layer.h"

namespace BinFunc
{
    class Convolution ;
    class BatchNorm ;
    class SumPooling ;
    class MaxPooling ;
    class Quantize ;
} ;

class BinFunc::Convolution
{
    public:
        //functions
        Convolution(uint32_t out_depth, tConvParams* in_params) ;
        tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tMultiBit* execute(tBit* p_inputs) ;

        //weight/code exportings
        void extract_filter_bias(tMultiBit* p_bias) ;
        void extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias) ;
        void export_weights(FILE* fd) ;

        //information/debugging
        void get_outhw(tRectangle* ret_dim) ;
        void get_outdep(uint32_t* ret_dep) ;

    private:
        static const uint16_t NUM_ADDS_K = 2 ;
        //inline helper functions
        uint64_t inline get_input_i(uint32_t ph, uint32_t pw, uint32_t di) ;
        uint64_t inline get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od) ;
        uint64_t inline get_output_i(uint32_t ph, uint32_t pw, uint32_t od) ;
        bool inline retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
            uint32_t* di, uint32_t* fh, uint32_t* fw) ;

        //weights
        uint8_t* p_filters ;
        uint8_t* p_tern ;
        //parameters
        bool b_prep ;
        uint64_t flen ;
        tDimensions lay_dim ;
        uint32_t OutDepth ;
        tConvParams conv ;
        tRectangle out_hw ;
        tRectangle offset_window ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

class BinFunc::BatchNorm
{
    public:
        //functions
        BatchNorm(tBNormParams* in_params) ;
        tDimensions* prep(tDimensions* ret_dim) ;

        //weight/code exportings
        void extract_bias(FILE* fd_filt, tMultiBit* p_bias, tMultiBit* p_slope) ;

    private:
        //parameters
        tBNormParams bparams ;
        tDimensions lay_dim ;
} ;

class BinFunc::SumPooling
{
    public:
        //functions
        SumPooling(tPoolParams* in_params) ;
        tDimensions* prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tMultiBit* execute(tMultiBit* p_inputs) ;

        //weight/code exportings
        void extract_bias(tMultiBit* p_bias) ;

        //nformation/debugging
        void get_outhw(tRectangle* ret_dim) ;
        void get_outdep(uint32_t* ret_dep) ;

    private:
        //inline helper functions
        uint64_t inline get_input_i(uint32_t ph, uint32_t pw, uint32_t di) ;
        uint64_t inline get_output_i(uint32_t ph, uint32_t pw, uint32_t od) ;
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        tRectangle offset_window ;
        tPoolParams pool ;
        tRectangle out_hw ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

class BinFunc::MaxPooling
{
    public:
        //functions
        MaxPooling(tPoolParams* in_params) ;
        tDimensions* prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tBit* execute(tBit* p_inputs) ;
    private:
        //inline helper functions
        uint64_t inline get_input_i(uint32_t ph, uint32_t pw, uint32_t di) ;
        uint64_t inline get_output_i(uint32_t ph, uint32_t pw, uint32_t od) ;
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        tRectangle offset_window ;
        tPoolParams pool ;
        tRectangle out_hw ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

class BinFunc::Quantize
{
    public:
        //functions
        Quantize(tQParams* qparam) ;
        #ifdef ENCRYPTED
        tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, uint32_t* p_slope, TFheGateBootstrappingCloudKeySet* in_bk) ;
        #else
        tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, tMultiBit* p_slope, TFheGateBootstrappingCloudKeySet* in_bk) ;
        #endif
        tBit* execute(tMultiBit* p_inputs, tMultiBit* p_bias) ;
        tMultiBit* add_bias(tMultiBit* p_inputs, tMultiBit* p_bias) ;
        #ifdef ENCRYPTED
        tFixedPoint* relu_shift(tMultiBit* p_inputs, tMultiBit* p_bias, uint32_t* p_slope);
        #else
        tFixedPoint* relu_shift(tMultiBit* p_inputs, tMultiBit* p_bias, tMultiBit* p_slope);
        #endif

        //weight/code exportings
        #ifndef ENCRYPTED
        void extract_bias(tMultiBit* p_bias, tMultiBit* p_slope) ;
        void export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope) ;
        #endif

    private:
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        uint32_t dim_len ;
	    uint8_t shift_bits ;
	    uint8_t slope_bits ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

#endif

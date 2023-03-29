/***********************************************************************
* FILENAME :        IntFunc.h
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file contains all the Integer input layer functions for a
*       a nerual network.
*
* PUBLIC CLASSES :
*       class IntFunc::Convolution Class
*       class IntFunc::BatchNorm
*       class IntFunc::SumPooling
*       class IntFunc::Quantize
*               
* NOTES :
*
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/

#ifndef _INT_FUNC_H_
#define _INT_FUNC_H_

#include <cstdio>
#include "Layer.h"
namespace IntFunc
{
    class Convolution
    {
        public:
            //functions
            Convolution(uint16_t out_depth, tConvParams* in_params) ;
            tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tFixedPoint* execute(tFixedPoint* p_inputs) ;

            //weight convert functions
            void extract_filter_bias(tMultiBit* p_bias) ;
            void extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias) ;
            void export_weights(FILE* fd) ;

        private:
            bool inline retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
                                      uint32_t* di, uint32_t* fh, uint32_t* fw) ;
            uint64_t inline get_input_i(uint32_t ph, uint32_t pw, uint32_t di) ;
            uint64_t inline get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od) ;
            uint64_t inline get_output_i(uint32_t ph, uint32_t pw, uint32_t od) ;
        	//weights
            uint8_t* p_filters ;
            uint8_t* p_tern ;
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            uint32_t flen ;
            uint32_t in_up_bound ;
            uint32_t OutDepth ;
            tConvParams conv ;
            tRectangle out_hw ;
            tRectangle offset_window ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;

    class BatchNorm
    {
        public:
            //functions
            BatchNorm(tBNormParams* in_params) ;
            tDimensions* prep(tDimensions* ret_dim, uint8_t bit_scale) ;

            //weight/code exportings
            void extract_bias(FILE* fd_filt, tMultiBit* p_bias, tMultiBit* p_slope) ;

        private:
            //parameters
            tBNormParams bparams ;
            tDimensions lay_dim ;
            uint32_t scale ;
    } ;


    class SumPooling
    {
        public:
            //functions
            SumPooling(tPoolParams* in_params) ;
            tDimensions* prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tFixedPoint* execute(tFixedPoint* p_inputs) ;

            //weight convert functions
            void extract_bias(tMultiBit* p_bias) ;

        private:
            //inline helper functions
            uint64_t inline get_input_i(uint32_t ph, uint32_t pw, uint32_t di) ;
            uint64_t inline get_output_i(uint32_t ph, uint32_t pw, uint32_t od) ;
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            tPoolParams pool ;
            tRectangle out_hw ;
            tRectangle offset_window ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;

    class Quantize
    {
        public:
            //functions
            Quantize(tQParams* qparam) ;
            #ifndef ENCRYPTED
            tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, tMultiBit* p_slope, 
                TFheGateBootstrappingCloudKeySet* in_bk) ;
            #else
            tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit* p_bias, uint32_t* p_slope, TFheGateBootstrappingCloudKeySet* in_bk) ;
            #endif
            tBit* execute(tFixedPoint* p_inputs, tMultiBit* p_bias) ;
            tFixedPoint* add_bias(tFixedPoint* p_inputs, tMultiBit* p_bias) ;
            #ifndef ENCRYPTED
            tFixedPoint* relu_shift(tFixedPoint* p_inputs, tMultiBit* p_bias, tMultiBit* p_slope) ;
            #else 
            tFixedPoint* relu_shift(tFixedPoint* p_inputs, tMultiBit* p_bias, uint32_t* p_slope) ;
            #endif

            //weight convert functions
            #ifndef ENCRYPTED
            void extract_bias(tMultiBit* p_bias, tMultiBit* p_slope) ;
            void export_weights(FILE* fd, tMultiBit* p_bias, tMultiBit* p_slope) ;
            #endif

        private:
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            uint32_t dim_len ;
            uint32_t add_offset ;
            uint8_t shift_bits ;
            uint8_t slope_bits ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;
}

#endif

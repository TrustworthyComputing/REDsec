#ifndef _INT_FUNC_H_
#define _INT_FUNC_H_

#include <cstdio>
#include"Layer.h"
namespace IntFunc
{
    class Convolution
    {
        public:
            //functions
            Convolution(uint16_t out_depth, tConvParams* in_params) ;
            tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tMultiBit* execute(tMultiBit* p_inputs) ;

            //weight convert functions
            void extract_filter_bias(tMultiBit* p_bias) ;
            void extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias) ;
            void export_weights(FILE* fd) ;

            //weights
            tBit* p_filters ; // bit array
            tBit* p_bias ; // bit array
            bool b_prep ;

        private:
            //parameters
            tDimensions lay_dim ;
            uint32_t OutDepth ;
            tConvParams conv ;
            tRectangle out_hw ;
            tRectangle offset_window ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;

    class SumPooling
    {
        public:
            //functions
            SumPooling(tPoolParams* in_params) ;
            tDimensions* prep(tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tMultiBit* execute(tMultiBit* p_inputs) ;

            //weight convert functions
            void extract_bias(tMultiBit* p_bias) ;

        private:
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            tPoolParams pool ;
            tRectangle out_hw ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;

    class Quantize
    {
        public:
            //functions
            Quantize() ;
            tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit** p_bias, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tBit* execute(tMultiBit* p_inputs, tMultiBit* p_bias) ; // output is bit array

            //weight convert functions
            void extract_bias(tMultiBit* p_bias) ;
            void export_weights(FILE* fd, tMultiBit* p_bias) ;

            tMultiBit* add_offset_enc ;

        private:
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            uint32_t dim_len ;
            uint32_t add_offset ;
            TFheGateBootstrappingCloudKeySet* bk ;
    } ;
}

#endif

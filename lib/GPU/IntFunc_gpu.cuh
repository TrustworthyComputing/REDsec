#ifndef _INT_FUNC_H_
#define _INT_FUNC_H_

#include <cstdio>
#include "Layer.cuh"
#include "REDcuFHE/redcufhe_gpu.cuh"
namespace IntFunc
{
    class Convolution
    {
        public:
            //functions
            Convolution(uint16_t out_depth, tConvParams* in_params) ;
            tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim) ;
            tFixedPointPacked* execute(tFixedPointPacked* p_inputs) ;

            //weight convert functions
            void extract_filter_bias(tFixedPointPacked* p_bias) ;
            void extract_bias(FILE* fd_filt, tMultiBitPacked* p_bias, uint16_t* p_slope) ;
            void export_weights(FILE* fd) ;
            tFixedPointPacked* p_output;

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
            uint32_t flen;
            uint32_t in_up_bound;
            uint32_t OutDepth ;
            tConvParams conv ;
            tRectangle out_hw ;
            tRectangle offset_window ;
    } ;

    class BatchNorm
    {
        public:
            //functions
            BatchNorm(tBNormParams* in_params) ;
            tDimensions* prep(tDimensions* ret_dim, uint8_t bit_scale) ;

            //weight/code exportings
            void extract_bias(FILE* fd_filt, tFixedPointPacked* p_bias) ;

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
            tDimensions* prep(tDimensions* ret_dim) ;
            tFixedPointPacked* execute(tFixedPointPacked* p_inputs) ;

            //weight convert functions
            void extract_bias(tFixedPointPacked* p_bias) ;
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
    } ;

    class Quantize
    {
        public:
            //functions
            Quantize(tQParams* qparam) ;
            tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBitPacked* p_bias, uint16_t* p_slope);
            tBitPacked* execute(tFixedPointPacked* p_inputs, tFixedPointPacked* p_bias) ;
            tFixedPointPacked* add_bias(tFixedPointPacked* p_inputs, tMultiBitPacked* p_bias) ;
            tFixedPointPacked* relu_shift(tFixedPointPacked* p_inputs, tMultiBitPacked* p_bias, uint16_t* p_slope) ;

            //weight convert functions
            void extract_bias(tMultiBitPacked* p_bias, uint16_t* p_slope) ;
            void export_weights(FILE* fd, tMultiBitPacked* p_bias, uint16_t* p_slope) ;

            tBitPacked* p_output;
            tFixedPointPacked* x_add;
            tFixedPointPacked* x_bn;
            tFixedPointPacked* p_output_relu;

        private:
            //parameters
            bool b_prep ;
            tDimensions lay_dim ;
            uint32_t dim_len ;
            uint32_t add_offset ;
            uint8_t shift_bits ;
            uint8_t slope_bits ;
    } ;
}

#endif

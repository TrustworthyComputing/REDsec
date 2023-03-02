#ifndef _BIN_FUNC_H_
#define _BIN_FUNC_H_

#include <cstdio>
#include "Layer.cuh"
#include "REDcuFHE/redcufhe_gpu.cuh"
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
        tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim) ;
        tMultiBitPacked* execute(tBitPacked* p_inputs) ;

        //weight/code exportings
        void extract_filter_bias(tMultiBitPacked* p_bias) ;
        void extract_bias(FILE* fd_filt, tMultiBitPacked* p_bias, eBiasType e_bias) ;
	    void export_weights(FILE* fd) ;

        //debugging
        void get_outhw(tRectangle* ret_dim) ;
        void get_outdep(uint32_t* ret_dep) ;

        tMultiBitPacked* p_window;
        tBitPacked* p_inputs_bar;
        tMultiBitPacked* p_output;

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
} ;

class BinFunc::BatchNorm
{
    public:
        //functions
        BatchNorm(tBNormParams* in_params) ;
        tDimensions* prep(tDimensions* ret_dim) ;

        //weight/code exportings
        void extract_bias(FILE* fd_filt, tMultiBitPacked* p_bias, uint16_t* p_slope) ;

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
        tDimensions* prep(tDimensions* ret_dim) ;
        tMultiBitPacked* execute(tMultiBitPacked* p_inputs) ;

        //weight/code exportings
        void extract_bias(tMultiBitPacked* p_bias) ;

        //debuggging
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
} ;

class BinFunc::MaxPooling
{
    public:
        //functions
        MaxPooling(tPoolParams* in_params) ;
        tDimensions* prep(tDimensions* ret_dim) ;
        tBitPacked* execute(tBitPacked* p_inputs) ;
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
} ;

class BinFunc::Quantize
{
    public:
        //functions
        Quantize(tQParams* qparam) ;
        tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBitPacked* p_bias, uint16_t* p_slope) ;
        tBitPacked* execute(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias) ;
        tMultiBitPacked* add_bias(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias) ;
        tFixedPointPacked* relu_shift(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias, uint16_t* p_slope) ;

        //weight/code exportings
        void extract_bias(tMultiBitPacked* p_bias, uint16_t* p_slope) ;
        void export_weights(FILE* fd, tMultiBitPacked* p_bias, uint16_t* p_slope) ;

        tBitPacked* p_output ;
        tFixedPointPacked* p_output_relu ;
        tMultiBitPacked* x_bn ;
        tFixedPointPacked* x_fp ;
        tMultiBitPacked* x_add ;
      

    private:
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        uint32_t dim_len ;
 	    uint8_t shift_bits ;
        uint8_t slope_bits ;
} ;

#endif

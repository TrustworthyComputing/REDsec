#ifndef _BIN_FUNC_H_
#define _BIN_FUNC_H_

#include <cstdio>
#include "Layer.h"

namespace BinFunc
{
    class Convolution ;
    class SumPooling ;
    class MaxPooling ;
    class Quantize ;
} ;

class BinFunc::Convolution
{
    public:
        //weights
        tBit* p_filters ;
        tBit* p_bias ;
        //functions
        Convolution(uint16_t out_depth, tConvParams* in_params) ;
        tDimensions* prep(FILE* fd_filt, tDimensions* ret_dim, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tMultiBit* execute(tBit* p_inputs) ;

        //weight/code exportings
        void extract_bias(FILE* fd_filt, tMultiBit* p_bias, eBiasType e_bias) ;
        void export_weights(FILE* fd) ;

        //debugging
        void get_outhw(tRectangle* ret_dim) ;
        void get_outdep(uint32_t* ret_dep) ;

    private:
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        uint32_t OutDepth ;
        tConvParams conv ;
        tRectangle offset_window ;
        tRectangle out_hw ;
        TFheGateBootstrappingCloudKeySet* bk ;
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

        //debuggging
        void get_outhw(tRectangle* ret_dim) ;
        void get_outdep(uint32_t* ret_dep) ;

    private:
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
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
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        tPoolParams pool ;
        tRectangle out_hw ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

class BinFunc::Quantize
{
    public:
        //functions
        Quantize() ;
        tDimensions* prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBit** p_bias, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tBit* execute(tMultiBit* p_inputs, tMultiBit* p_bias) ;

        //weight/code exportings
        void extract_bias(tMultiBit* p_bias) ;
        void export_weights(FILE* fd, tMultiBit* p_bias) ;

    private:
        //parameters
        bool b_prep ;
        tDimensions lay_dim ;
        uint32_t dim_len ;
        tMultiBit* add_offset_enc ;
        uint32_t add_offset ;
        TFheGateBootstrappingCloudKeySet* bk ;
} ;

#endif
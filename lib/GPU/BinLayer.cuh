#ifndef _LAYB_H_
#define _LAYB_H_

#include <cstdlib>
#include <cstdbool>
#include <cstdio>
#include "Layer.cuh"
#include "BinFunc_gpu.cuh"

typedef union _POOLSB
{
    BinFunc::MaxPooling* m ;
    BinFunc::SumPooling* s ;
} tPoolb ;

class BinLayer
{
    public:
        BinLayer(eConvType ec, uint16_t dep,
            ePoolType ep,
            eQuantType eq,
            tNetParams* np) ;
        tDimensions* prep(FILE* fd, tDimensions* dim) ;
        tBitPacked* execute(tBitPacked* p_in) ;

        //weight export
        void export_weights(FILE* fd_export) ;

        //debugging
        void set_print_layer(uint8_t i) ;

        //copies only; actual dimensions stored in layers
        tDimensions in_dim ;
        tDimensions out_dim ;

    private:
        void  set_version(uint8_t version, tNetParams* net) ;
        void* run(eAction e_act, tActParams* ap, FILE* fd) ;
#ifdef _PRINT_LAYER_
        void print_layer_out(tBit* p_indat, tDimensions* dim) ;
        void print_layer_mid(tMultiBit* p_indat, tDimensions* dim) ;
        char print_label[10] = {0} ;
#endif
        tMultiBitPacked* p_bias ;
        uint16_t* p_slope ;
        //state variables
        bool b_prep = false ;
        bool b_print_layer = false ;
	    FILE* fdebug = NULL ;
        //configurations
        uint16_t conv_depth ;
        eConvType e_conv ;
        eBiasType e_bias ;
        ePoolType e_pool ;
        eQuantType e_activation ;
        BinFunc::Convolution* lconv ;
        tPoolb lpool ;
        BinFunc::Quantize* lquant ;
#ifdef _WEIGHT_CONVERT_
        BinFunc::BatchNorm*  bnorm ;
#endif
} ;

#endif

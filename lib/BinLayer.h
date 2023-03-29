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
**************************************************************************************************/

#ifndef _LAYB_H_
#define _LAYB_H_

#include <cstdlib>
#include <cstdbool>
#include <cstdio>
#include "Layer.h"

#if defined(ENCRYPTED)
#include "BinFunc_enc.h"
#else
#include "BinFunc.h"
#endif

typedef union _POOLSB
{
    BinFunc::MaxPooling* m ;
    BinFunc::SumPooling* s ;
} tPoolb ;

class BinLayer
{
    public:
        BinLayer(eConvType ec, uint16_t dep, ePoolType ep, eQuantType eq,
                 tNetParams* np, TFheGateBootstrappingCloudKeySet* in_bk) ;
        tDimensions* prep(FILE* fd, tDimensions* dim) ;
        void* execute(tBit* p_in) ;

        //weight export
        void export_weights(FILE* fd_export) ;

        //copies only; actual dimensions stored in layers
        tDimensions in_dim ;
        tDimensions out_dim ;

    private:
        void  set_version(uint8_t version, tNetParams* net) ;
        void* run(eAction e_act, tActParams* ap, FILE* fd) ;
        tMultiBit* p_bias ;
#ifdef ENCRYPTED
        uint32_t* p_slope ;
#else
        tMultiBit* p_slope ;
#endif
        TFheGateBootstrappingCloudKeySet* bk ;
        //state variables
        bool b_prep = false ;
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

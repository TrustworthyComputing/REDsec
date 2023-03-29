/**************************************************************************************************
* FILENAME :        IntLayer.h
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file pieces together the integer Convolution/Fully Connected, Batch Norm, Pooling and 
*       Activation functions in the optimal order for TFHE REDsec. 
*
* NOTES :     
* AUTHOR :    Lars Folkerts
* START DATE :    16 Aug 20
*************************************************************************************************/

#ifndef _LAYD_H_
#define _LAYD_H_
#include <stdio.h>
#include <stdbool.h>
#include "Layer.h"

#if defined(ENCRYPTED)
#include "BinFunc_enc.h"
#include "IntFunc_enc.h"
#else
#include "BinFunc.h"
#include "IntFunc.h"
#endif

typedef union _POOLSD
{
    BinFunc::MaxPooling* m ;
    IntFunc::SumPooling* s ;
} tPoold ;

class IntLayer
{
        public:
            //Constructor
            IntLayer(eConvType ec, uint16_t dep, ePoolType ep, eQuantType eq,
                     tNetParams* np, TFheGateBootstrappingCloudKeySet* in_bk) ;
            tDimensions* prep(FILE* fd, tDimensions* dim) ;
            #if defined(ENCRYPTED)
            void* execute(tMultiBit* p_in);
            #else
            void* execute(tFixedPoint* p_in) ;
            #endif

            //weight convert
            void export_weights(FILE* fd) ;

            tDimensions in_dim ;
            tDimensions out_dim ;

        private:
            void set_version(uint8_t v, tNetParams* net) ;
            //function to execute function in proper order
            void* run(eAction e_act, tActParams* ap, FILE* fd) ;
            tMultiBit* p_bias ;
            #ifndef ENCRYPTED
            tMultiBit* p_slope ;
            #else
            uint32_t* p_slope;
            #endif
            //state variables
            TFheGateBootstrappingCloudKeySet* bk ;
            bool b_initialized = false ;
	    FILE* fdebug = NULL ;
            //configurations
            eConvType e_conv ;
            eBiasType e_bias ;
            ePoolType e_pool ;
            eQuantType e_activation ;
	    uint32_t conv_depth ;
            IntFunc::Convolution* lconv ;
            tPoold lpool ;
            IntFunc::Quantize* lquant ;
#ifdef _WEIGHT_CONVERT_
            IntFunc::BatchNorm*  bnorm ;
#endif
} ;

#endif

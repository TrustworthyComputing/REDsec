#ifndef _LAYD_H_
#define _LAYD_H_
#include <stdio.h>
#include <stdbool.h>
#include "Layer.cuh"
#include "BinFunc_gpu.cuh"
#include "IntFunc_gpu.cuh"

/****************************** Structs And Enums *******************************/
typedef union _POOLSD
{
    BinFunc::MaxPooling* m ;
    IntFunc::SumPooling* s ;
} tPoold ;
/**************************** Function Declarations *****************************/
class IntLayer
{
        public:
            //Constructor
            IntLayer(eConvType ec, uint16_t dep,
                ePoolType ep,
                eQuantType eq,
                tNetParams* np) ;
            tDimensions* prep(FILE* fd, tDimensions* dim) ;
            tBitPacked* execute(tMultiBitPacked* p_in);

            //weight convert
            void export_weights(FILE* fd) ;

            tDimensions in_dim ;
            tDimensions out_dim ;

	        void set_print_layer(uint8_t i) ;

        private:
            //function to execute function in proper order
            void* run(eAction e_act, tActParams* ap, FILE* fd) ;
            void set_version(uint8_t v, tNetParams* net) ;
            tFixedPointPacked* p_bias ;
            uint16_t* p_slope ;
            //state variables
            bool b_initialized = false ;
            bool b_print_layer = false ;
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
#ifdef _PRINT_LAYER_
            //debugging
            void print_layer_out(tFixedPoint* p_indat, tDimensions* dim) ;
            char print_label[10] = {0} ;
#endif
} ;

#endif

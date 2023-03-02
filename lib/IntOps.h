/*************************************************************************************************
* FILENAME :        IntOps.h
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file contains all the basic unencrypted integer operation functions.
*
* NOTES :     Unencrypted and Weight Convert only. Encrypeded interace is in a separate file
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/

#ifndef _INT_OPS_H_
#define _INT_OPS_H_

#include "Layer.h"
namespace IntOps
{
    void multiply(tFixedPoint* result, tFixedPoint* in1, tBit* in2, uint8_t in1_bits, void* bk) ;
    void multiply_pc_ints(tFixedPoint* result, tFixedPoint* in1, const tMultiBit* multicand, 
                          uint8_t in1_bits, uint8_t in2_bits, void* bk);
    void invert(tFixedPoint* result, tFixedPoint* in1, tBit* in2, 
	            uint8_t in1_bits, void* bk) ;
    void add(tFixedPoint* result, tFixedPoint* in1, tFixedPoint* in2, 
             uint8_t in1_bits, void* bk) ;
    void add_pc_ints(tFixedPoint* result, tFixedPoint* in1, 
                     const tMultiBit* addend, uint8_t in1_bits, void* bk) ;
    void binarize(tBit* result, tFixedPoint* in1,  uint8_t input_bits, void* bk) ;
    void shift(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits,
               uint8_t shift_bits, void* bk);
    void relu(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, void* bk);
}
#endif

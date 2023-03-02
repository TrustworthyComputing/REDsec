/*************************************************************************************************
* FILENAME :        IntOps.cpp
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
#ifndef ENCRYPTED
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdio>
#include "IntOps.h"

/*********************************************************************************
*    Func:      IntOps::multiply
*    Desc:      Multiplies (XNOR) a signed int with a bit
*    Inputs:    tFixedPoint* - result pointer
*               tFixedPoint* - first multicand
*               tBit* - second multicand
*               uint8_t - number of bits in frist multicand
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void IntOps::multiply(tFixedPoint* result, tFixedPoint* in1, tBit* in2, 
                      uint8_t in1_bits, void* bk)
{
    tFixedPoint x = *in1 ;
    tBit y = *in2 ;
    tFixedPoint inv = (~x) + 1 ;
    (*result) = (y==1)?x:inv ;
}

/*********************************************************************************
*    Func:      IntOps::multiply_pc_ints
*    Desc:      Multiplies two signed multibit integers together
*    Inputs:    tFixedPoint* - result pointer
*               tFixedPoint* - first multicand
*               tMultiBit* - second multicand
*               uint8_t - number of bits in first multicand
*               uint8_t - number of bits in second multicand
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void IntOps::multiply_pc_ints(tFixedPoint* result, tFixedPoint* in1, 
                              const tMultiBit* multiplicand, uint8_t in1_bits, 
                              uint8_t in2_bits, void* bk)
{
	//multiplicand is a in2-bit constant
    tFixedPoint x = *in1 ;
    (*result) = x* (*multiplicand) ; 
}

/*********************************************************************************
*    Func:      IntOps::invert
*    Desc:      Inverts ciphertext (NOT)
*    Inputs:    tFixedPoint* - result pointer
*               tFixedPoint* - first input
*               tBit* - 0 for as is, 1 for invert
*               uint8_t - number of bits in second multicand
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void IntOps::invert(tFixedPoint* result, tFixedPoint* in1, tBit* in2, 
	                uint8_t in1_bits, void* bk)
{
	tFixedPoint x = (*in1);
	//propagate y through all bits
	uint32_t y = (*in2==1)?(0xFFFFFFFF):0;
	//and crate a result mask
	uint32_t mask = 0xFFFFFFFF ; //(y>>(32-in1_bits)) ;
	//XNOR of bits
	(*result) = (~(x^y))&mask ;	    
}
/********************************************************v*************************
*    Func:      BinOps::add
*    Desc:      Addition of two encrypted signed integers
*    Inputs:    tFixedPoint* - result pointer
*               tFixedPoint* - first addend
*               uint8_t - number of bits of longest input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void IntOps::add(tFixedPoint* result, tFixedPoint* in1, tFixedPoint* in2,
                 uint8_t in1_bits, void* bk)
{
    tFixedPoint x = *in1 ;
    tFixedPoint y = *in2 ;
    *result = x+y ;
}

/*********************************************************************************
*    Func:      IntOps::add_pc_ints
*    Desc:      Addition of one plaintext and one ciphertext integer
*    Inputs:    tFixedPoint* - result pointer
*               tMultiBit* - first ciphertext addend
*               tMultiBit* - plaintext addend
*               uint8_t - number of bits of longest input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void IntOps::add_pc_ints(tFixedPoint* result, tFixedPoint* in1, const tMultiBit* addend, 
                         uint8_t in1_bits, void* bk)
{
    tFixedPoint x = *in1 ;
	//addend is a ptxt constant
    *result = x+(*addend) ;
}

/********************************************************************************
*    Func:      IntOps::binarize
*    Desc:      Variant of Sign Activation Function.
*               Performs bitshift to extract single bit.
*    Inputs:    tBit* - result pointer
*               tFixedPoint* - input integer
*               uint8_t - amount of shift
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void IntOps::binarize(tBit* result, tFixedPoint* in1, 
                      uint8_t input_bits, void* bk)
{
    //take the MSB (sign bit)
    tFixedPoint val = *in1 ;
    (*result) =  (val<0)?0:1 ;
}

/********************************************************************************
*    Func:      IntOps::relu
*    Desc:      Variant of ReLU Activation Function.
*               AND with the sign bit for ReLU.
*    Inputs:    tFixedPoint* - integer result
*               tFixedPoint* - input integer
*               uint8_t - amount of bits in input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void IntOps::relu(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, void* bk)
{
    //take the MSB (sign bit)
    tFixedPoint val = *in1 ;
#ifndef _WEIGHT_CONVERT_
    tBit inv_sign_bit = (val<0)? 0 : 1 ; //(~(val>>(in_bits-1))&0x1 ;
    val = (val >= (1<<input_bits)) ? (1<<input_bits)-1:val  ;
#else
    *result = (val < 0)?0:val ;
    return ;
#endif
	//AND each bit with MSB - only keep values greater than threshold
    *result = 0 ;
    for(uint8_t i = 0 ; i < input_bits; i++)
    {
#ifndef _WEIGHT_CONVERT_
        (*result) |= (val & (tMultiBit)inv_sign_bit << i) ;
#endif
    }
}

/********************************************************************************
*    Func:      IntOps::shift
*    Desc:      Variant of ReLU Activation Function.
*               Shift to discretize.
*    Inputs:    tFixedPoint* - integer result
*               tFixedPoint* - input integer
*               uint8_t - amount of bits in input
*               uint8_t - amount of shift
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void IntOps::shift(tFixedPoint* result, tFixedPoint* in1, 
                   uint8_t input_bits, uint8_t shift_bits, void* bk)
{
    assert((input_bits>0) && (shift_bits <= input_bits));
    //take the MSB (sign bit)
    tFixedPoint val = *in1 ;
    //shift over result to keep values small
#ifndef _WEIGHT_CONVERT_
   (*result) = (*in1)>>(shift_bits) ; 
#else
   *result /= (1<<(shift_bits)) ;
#endif
}

#endif //!ENCRYPTED

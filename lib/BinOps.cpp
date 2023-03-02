/**************************************************************************************************
* FILENAME :        BinOps.cpp
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       This file contains all the basic binary unencrypted operation functions.
*
* NOTES :     Unencrypted and Weight Convert only. Encrypeded interace is in a separate file
* AUTHOR :    Lars Folkerts, Charles Gouert
* START DATE :    16 Aug 20
*************************************************************************************************/
#ifndef ENCRYPTED
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <iostream>
#ifdef _WEIGHT_CONVERT_
#include <cmath>
#endif
using namespace std;
#include "BinOps.h"
#define ASSERT_BIT
typedef uint8_t bitpack_t ;

typedef enum _DATA_FORMAT
{
    NULL_FMT,
    BIN_FMT,
    TERN_FMT,
    UINT32_FMT,
    INT32_FMT,
    NUM_FMT
} eFmt ;

/*********************************************************************************
*    Func:      BinOps::multiply
*    Desc:      Multiplies (XNOR) of two bits
*    Inputs:    tBit* - result pointer
*               tBit* - first multicand 
*               tBit* - second multicand
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::multiply(tBit* result, tBit* in1, tBit* in2, void* bk)
{
    tBit x = *in1 ;
    tBit y = *in2 ;
#ifdef ASSERT_BIT
    assert((x == 0) ||(x == 1)) ;
    assert((y == 0) ||(y == 1)) ;
#endif
#ifndef _WEIGHT_CONVERT_
    (*result) = (~(x^y)) & 0x1 ; //XNOR of LSB
#else
    (*result) =  x*y ;
#endif
}

/*********************************************************************************
*    Func:      BinOps::multiply_pc_ints
*    Desc:      Multiplies two unsigned multibit integers together
*    Inputs:    tMultiBit* - result pointer
*               tMultiBit* - first multicand 
*               tMultiBit* - second multicand
*               uint8_t - number of bits in first multicand
*               uint8_t - number of bits in second multicand
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::multiply_pc_ints(tMultiBit* result, tMultiBit* in1, const tMultiBit* multicand, uint8_t in1_bits, uint8_t in2_bits, void* bk)
{
    tBit x = *in1 ;
    (*result) =  x*(*multicand) ;
}

/*********************************************************************************
*    Func:      BinOps::add_bit
*    Desc:      Addition of two bits
*    Inputs:    tMultiBit* - result pointer
*               tBit* - first addend 
*               tBit* - second addend
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::add_bit(tMultiBit* result, tBit* in1, tBit* in2, void* bk)
{
#ifdef ASSERT_BIT
    assert((*in1 == 0) ||(*in1 == 1)) ;
    assert((*in2 == 0) ||(*in2 == 1)) ;
#endif
#ifndef _WEIGHT_CONVERT_
    tMultiBit x = (*in1) & 0x1 ;
    tMultiBit y = (*in2) & 0x1 ;
    *result = x+y ;
#else
    *result = (*in1)+(*in2) ;
#endif
}

/********************************************************v*************************
*    Func:      BinOps::add
*    Desc:      Addition of two encrypted unsigned integers
*    Inputs:    tMultiBit* - result pointer
*               tMultiBit* - first addend
*               tMultiBit* - second addend
*               uint8_t - number of bits of longest input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::add(tMultiBit* result, tMultiBit* in1, tMultiBit* in2,
    uint8_t in_bits, void* bk)
{
    assert((result != NULL) && (in1 != NULL) && (in2 != NULL)) ;
    tMultiBit mask = ~(0xFFFFFFFF << in_bits) ;
#ifndef _WEIGHT_CONVERT_
    tMultiBit x = *in1 ; //x &= mask ;
    tMultiBit y = *in2 ; //y &= mask ;
    *result = x+y ;
#else
    *result = (*in1)+(*in2) ;
#endif
}

/*********************************************************************************
*    Func:      BinOps::add_pc_ints
*    Desc:      Addition of one plaintext and one ciphertext integer
*    Inputs:    tFixedPoint* - result pointer
*               tMultiBit* - first ciphertext addend
*               tMultiBit* - plaintext addend
*               uint8_t - number of bits of longest input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::add_pc_ints(tFixedPoint* result, tMultiBit* in1, const tMultiBit* addend,
    uint8_t in_bits, void* bk)
{
    assert((result != NULL) && (in1 != NULL)) ;
    *result = (*in1)+(*addend) ;
}

/*********************************************************************************
*    Func:      BinOps::inc
*    Desc:      Addition of one integer with a bit
*    Inputs:    tMultiBit* - result pointer
*               tMultiBit* - first integer addend
*               tBit* - bit addend
*               uint8_t - number of bits of integer input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::inc(tMultiBit* result, tMultiBit* in1, tBit* in2, uint8_t in1_bits, void* bk)
{
    tMultiBit x = *in1 ;
    tBit y = *in2 ;
#ifdef ASSERT_BIT
    assert((y == 0) ||(y == 1)) ;
#endif
    *result = x+y ; //add 0 or 1
}

/********************************************************************************
*    Func:      BinOps::max
*    Desc:      Bitwise max (OR) function
*    Inputs:    tBit* - result pointer
*               tBit* - first input
*               tBit* - second input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation
*********************************************************************************/
void BinOps::max(tBit* result, tBit* in1, tBit* in2, void* bk)
{
    tBit x = *in1 ;
    tBit y = *in2 ;
#ifdef ASSERT_BIT
    assert((x == 0) ||(x == 1)) ;
    assert((y == 0) ||(y == 1)) ;
#endif
#ifndef _WEIGHT_CONVERT_
    (*result) = x|y ; //Max is logical OR
#else
    *result = (x>y)?x:y ;
#endif
}

/********************************************************************************
*    Func:      BinOps::binarize
*    Desc:      Variant of Sign Activation Function. 
*    		Performs bitshift to extract single bit. 
*    Inputs:    tBit* - result pointer
*               tMultiBit* - input integer
*               uint8_t - amount of shift
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void BinOps::binarize(tBit* result, tMultiBit* in1, uint8_t in_bits, void* bk)
{
    int val = (int) *in1 ;
#ifdef ZERO_BRIDGE
    tMultiBit mask = (1<<(in_bits-1)) ;
#else
    int mask = 0 ;
#endif
    //take MSB
    (*result) =  (tBit)((val<mask)?0:1);
}

/********************************************************************************
*    Func:      BinOps::relu
*    Desc:      Variant of ReLU Activation Function.
*               AND with the sign bit for ReLU.
*    Inputs:    tFixedPoint* - integer result
*               tMultiBit* - input integer
*               uint8_t - amount of bits in input
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void BinOps::relu(tFixedPoint* result, tMultiBit* in1, uint8_t in_bits,  void* bk)
{
    tMultiBit val = *in1 ;
    //take MSB
#ifndef _WEIGHT_CONVERT_
    tBit inv_sign_bit = (val>>(in_bits-1))&0x1 ; 
#else
    *result = (val < (in_bits<<1))?0:val ; return ;
#endif
	//AND each bit with MSB - only keep values greater than threshold
    *result = 0 ;
    //do not set top bit, recenter around 0
    for(uint8_t i = 0 ; i < in_bits-1; i++)
    {
#ifndef _WEIGHT_CONVERT_
        *result |= (val & (tMultiBit)inv_sign_bit << i) ;
#else
#endif
    }
}

/********************************************************************************
*    Func:      BinOps::shift
*    Desc:      Variant of ReLU Activation Function.
*               Shift to discretize.
*    Inputs:    tMultiBit* - integer result
*               tMultiBit* - input integer
*               uint8_t - amount of bits in input
*               uint8_t - amount of shift
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Only for use in unencrypted computation.
*               Must be centered around power of two.
*********************************************************************************/
void BinOps::shift(tMultiBit* result, tMultiBit* in1, uint8_t in_bits, uint8_t shift_bits, void* bk)
{
    assert((in_bits>0) && (shift_bits <= in_bits));
    //shift over result to keep values small
#ifndef _WEIGHT_CONVERT_
   *(result) = (*in1)>>(in_bits - shift_bits) ;
#else
   *result /= (1<<(in_bits - shift_bits+1)) ;
#endif
}
#ifndef _WEIGHT_CONVERT_

/********************************************************************************
*    Func:      BinOps::get_ternfilters
*    Desc:      Retrieves ternary weights from compressed file (!_WEIGHT_CONVERT_)
*    Inputs:    FILE* - input weights file (!_WEIGHT_CONVERT_)
*             	uint8_t* - returned binary (sign) bit (-1 or 1)
*               uint8_t* - returned ternary bit (TRUE if w=0)
*               uint32_t - number of weights to extract
*               float - ternary threshold (unused, for weight conversion)
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Compressed weights file only.
*********************************************************************************/
void BinOps::get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, float thresh, void* bk)
{
	//check data format
    uint8_t version = NULL_FMT ;
    size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in) ;
    assert(version == BIN_FMT || version == TERN_FMT) ;
    int NBITS = (version==BIN_FMT)?1:2 ;
    bool b_tern = (version == TERN_FMT) && (p_filt_tern != NULL) ;
    
    static const uint16_t bits = sizeof(bitpack_t)*8 ;
    uint32_t adj_len = (len*NBITS+bits-1)/bits ;
    bitpack_t* p_filt_pack = (bitpack_t*) calloc(adj_len, sizeof(bitpack_t)) ;

    sread = fread((void*)(p_filt_pack), sizeof(bitpack_t), adj_len, fd_in) ;
    for(uint32_t i = 0 ; i<adj_len ; i++)
    {   //unpack bits
    	for(int32_t j = 0 ;  j <= bits-1 ; j+=NBITS)
    	{
	        if((i*bits+j)/NBITS >= len) { continue ; }
    	    p_filt_b[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-1)) & 0x1) ;
 	        if(b_tern){ p_filt_tern[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-2)) & 0x1) ; }
    	}
    }
    if((version != TERN_FMT) && (p_filt_tern != NULL)){ memset(p_filt_tern, 0, len) ; }
    free(p_filt_pack) ;
}

/********************************************************************************
*    Func:      BinOps::get_intfilters
*    Desc:      Retrieves integer weights from file (!_WEIGHT_CONVERT_)
*    Inputs:    FILE* - input compressed weights file
*               tMultiBit* - returned integer filter (TRUE if w = 0)
*               uint32_t - number of weights to extract
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Compressed weights file only.
*********************************************************************************/
void BinOps::get_intfilters(FILE* fd_in, tMultiBit* p_filt_mb, uint32_t len, void* bk)
{
	//check data format
    uint8_t version = NULL_FMT ;
    size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in) ;
    assert((version == UINT32_FMT) || (version==INT32_FMT)) ;
    sread = fread((uint32_t*)(p_filt_mb), sizeof(uint32_t), len, fd_in) ;
}

#else //defined _WEIGHT_CONVERT_
/********************************************************************************
*    Func:      BinOps::get_ternfilters
*    Desc:      Retrieves ternary weights from file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - input Tensorflow weights file
*             	uint8_t* - returned binary (sign) bit (-1 or 1)
*               uint8_t* - returned ternary bit (TRUE if w = 0)
*               uint32_t - number of weights to extract
*               float - ternary threshold (if |w|<thresh then w=0)
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Tensorflow file for weight processing
*********************************************************************************/
void BinOps::get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, float thresh, void* bk)
{
    float* p_filt_f = (float*) calloc(len, sizeof(float)) ;
    size_t size = fread((void*)(p_filt_f), sizeof(float), len, fd_in) ;
    for(uint32_t i = 0 ; i<len ; i++)
    {
        ((tBit*)(p_filt_b))[i] = (tBit)((((float*)(p_filt_f))[i]) > 0)?1:0 ;
        ((tBit*)(p_filt_tern))[i] = (tBit)(fabs(((float*)(p_filt_f))[i]) < thresh)?1:0 ;
    }
    free(p_filt_f) ;
}

/********************************************************************************
*    Func:      BinOps::get_intfilters
*    Desc:      Retrieves integer weights from file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - input Tensorflow weights file
*               tMultiBit* - returned integer filter (TRUE if w = 0)
*               uint32_t - number of weights to extract
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     Tensorflow file for weight processing
*********************************************************************************/
void BinOps::get_intfilters(FILE* fd_in, tMultiBit* p_filt_mb, uint32_t len, void* bk)
{
    float* p_filt_f = (float*) calloc(len, sizeof(float)) ;
    size_t size = fread((void*)(p_filt_f), sizeof(float), len, fd_in) ;

    for(uint32_t i = 0 ; i<len ; i++)
    {
        p_filt_mb[i] = (tMultiBit) p_filt_f[i] ;
    }
    free(p_filt_f) ;
}

/********************************************************************************
*    Func:      BinOps::export_tern
*    Desc:      Exports ternary weights to compressed file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - output weights file
*               uint8_t* - input binary (sign) bit (-1 or 1)
*               uint8_t* - input ternary bit (TRUE if w = 0)
*               uint32_t - number of weights to export
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     
*********************************************************************************/
void BinOps::export_tern(FILE* fd_out, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, void* bk)
{
    static const int NBITS = 2;
    static const uint16_t bits = sizeof(bitpack_t)*8 ;
    bitpack_t exp = 0 ;
    uint32_t adj_len = (len*NBITS+bits-1)/bits ;
    
    static const uint8_t version = TERN_FMT ;
    fwrite(&version, sizeof(uint8_t), 1, fd_out) ;
    for(uint32_t i = 0 ; i<adj_len ; i++)
    {
    	exp = 0 ;
    	//pack bits
    	for(uint32_t j = 0 ;  (j <= bits-NBITS) && ((i*bits+j)/NBITS < len) ; j+=NBITS)
    	{
		    exp <<= 1 ;
    		exp |= p_filt_b[(i*bits+j)/NBITS] ;
		    exp <<= 1 ;
    		exp |= p_filt_tern[(i*bits+j)/NBITS] ;
    	}
    	fwrite(&exp, sizeof(exp), 1, fd_out) ;
    }
}

/********************************************************************************
*    Func:      BinOps::export_mulbits
*    Desc:      Exports unsigned integer weights to compressed file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - output weights file
*               tMultiBit* - input unsigned integer weights
*               uint32_t - number of weights to export
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     
*********************************************************************************/
void BinOps::export_mulbits(FILE* fd_out, tMultiBit* p_filt, uint32_t len, void* bk)
{
    uint32_t* p_filt_int = (uint32_t*) calloc(len, sizeof(uint32_t)) ;
    static const uint8_t version = UINT32_FMT ;
    fwrite(&version, sizeof(uint8_t), 1, fd_out) ;
    for(uint32_t i = 0 ; i<len ; i++)
    {
        //Need floor function for bias to int (0 is positive)
        //e.g. float bias 1 - 1.25 = -.25 (negative)
        //when int bias 1 - 2 = -1 (negative)
        p_filt_int[i] = (int32_t) floorf(p_filt[i]) ; 
    }
    fwrite(p_filt_int, sizeof(uint32_t), len, fd_out) ;
    free(p_filt_int) ;
}

/********************************************************************************
*    Func:      BinOps::export_signedBias
*    Desc:      Exports signed integer weights to compressed file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - output weights file
*               tMultiBit* - input unsigned integer weights
*               uint32_t - number of weights to export
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:     
*********************************************************************************/
void BinOps::export_signedBias(FILE* fd_out, tMultiBit* p_filt, uint32_t len, void* bk)
{    
    int32_t* p_filt_int = (int32_t*) calloc(len, sizeof(uint32_t)) ;
    static const uint8_t version = INT32_FMT ;
    fwrite(&version, sizeof(uint8_t), 1, fd_out) ;
    for(uint32_t i = 0 ; i<len ; i++)
    {
        //Need floor function for bias to int (0 is positive)
        //e.g. float bias 1 - 1.25 = -.25 (negative)
        //when int bias 1 - 2 = -1 (negative)
        p_filt_int[i] = (int32_t) floorf(p_filt[i]) ;
    }
    fwrite(p_filt_int, sizeof(int32_t), len, fd_out) ;
    free(p_filt_int) ;
}

/********************************************************************************
*    Func:      BinOps::export_fixedPoint
*    Desc:      Exports signed integer weights to compressed file (_WEIGHT_CONVERT_)
*    Inputs:    FILE* - output weights file
*               tMultiBit* - input unsigned integer weights
*               uint32_t - number of weights to export
*               void* - bootstrapping key (NULL, for compatibility)
*    Return:    None
*    Notes:
*********************************************************************************/
void BinOps::export_fixpt(FILE* fd_out, tFixedPoint* p_filt, uint32_t len, void* bk)
{
    int32_t* p_filt_int = (int32_t*) calloc(len, sizeof(uint32_t)) ;
    static const uint8_t version = INT32_FMT ;
    fwrite(&version, sizeof(uint8_t), 1, fd_out) ;
    for(uint32_t i = 0 ; i<len ; i++)
    {
        //Need floor function for bias to int (0 is positive)
        //e.g. float bias 1 - 1.25 = -.25 (negative)
        //when int bias 1 - 2 = -1 (negative)
        p_filt_int[i] = (int32_t) floorf(p_filt[i]) ; 
    }
    fwrite(p_filt_int, sizeof(int32_t), len, fd_out) ;
    free(p_filt_int) ;
}
#endif //WEIGHT_CONVERT
#endif //!ENCRYPTED

/**************************************************************************************************
* FILENAME :        BinOps.h
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

#ifndef _BIN_OPS_H_
#define _BIN_OPS_H_

#include <cstdio>
#include "Layer.h"

namespace BinOps
{
    void multiply(tBit* result, tBit* in1, tBit* in2, void* bk) ;
    void multiply_pc_ints(tMultiBit* result, tMultiBit* in1, const tMultiBit* multicand, uint8_t in1_bits, uint8_t in2_bits, void* bk) ;
    void add(tMultiBit* result, tMultiBit* in1, tMultiBit* in2, uint8_t in_bits, void* bk) ;
    void add_pc_ints(tFixedPoint* result, tMultiBit* in1, const tMultiBit* addend, uint8_t in_bits, void* bk) ;
    void add_bit(tMultiBit* result, tBit* in1, tBit* in2, void* bk) ;
    void inc(tMultiBit* result, tMultiBit* in1, tBit* in2, uint8_t in1_bits, void* bk) ;
    void max(tBit* result, tBit* in1, tBit* in2, void* bk) ;
    void binarize(tBit* result, tMultiBit* in1, uint8_t input_bits, void* bk) ;
    void shift(tMultiBit* result, tMultiBit* in1, uint8_t input_bits, uint8_t shift_bits, void* bk) ;
    void relu(tFixedPoint* result, tMultiBit* in1, uint8_t input_bits, void* bk) ;
    void get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, float thresh, void* bk) ;
    void get_intfilters(FILE* fd_in, tMultiBit* p_filt_b, uint32_t len, void* bk) ;
#ifdef _WEIGHT_CONVERT_
    //weight convert functions
    void export_tern(FILE* fd_out, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, void* bk) ;
    void export_mulbits(FILE* fd_out, tMultiBit* p_filt, uint32_t len, void* bk) ;
    void export_signedBias(FILE* fd_out, tMultiBit* p_filt, uint32_t len, void* bk) ;
    void export_fixpt(FILE* fd_out, tFixedPoint* p_filt, uint32_t len, void* bk) ;
#endif
} 
#endif

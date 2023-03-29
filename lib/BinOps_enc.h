#ifndef _BIN_OPS_H_
#define _BIN_OPS_H_

#include <cstdio>
#include "Layer.h"
#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
namespace BinOps
{
    // XNOR (1 bootstrap)
    void multiply(tBit* result, const tBit* a, const uint8_t b, TFheGateBootstrappingCloudKeySet* bk);
    void multiply_pc_ints(LweSample* result, LweSample* in1, const uint32_t* multicand, uint8_t in1_bits, uint8_t in2_bits, TFheGateBootstrappingCloudKeySet* bk);

    void add_bit(tMultiBit* result, const tBit* a, const tBit* b, TFheGateBootstrappingCloudKeySet* bk);
    
    // Unsigned Adder (3 * MAX(a->size,b->size) - 1 bootstraps)
    void add(tMultiBit* result, const tMultiBit* a, const tMultiBit* b, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);

    void add_int(LweSample* result, const LweSample* a, const LweSample* b, TFheGateBootstrappingCloudKeySet* bk);
    void add_pc_ints(LweSample* result, LweSample* in1, const uint16_t* addend, uint8_t in_bits, TFheGateBootstrappingCloudKeySet* bk) ;
    void add_int_inplace(LweSample* result, const LweSample* a,
    TFheGateBootstrappingCloudKeySet* bk);

    // Unsigned Incrementer (2 * a->size - 1 bootstraps)
    void inc(tMultiBit* result, const tMultiBit* a, const tBit* b, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);

    // Compute MAX(bit1, bit2) == OR(bit1, bit2)
    void max(tBit* result, const tBit* a, const tBit* b, TFheGateBootstrappingCloudKeySet* bk);

    // extract top bit of tMultiBit instance
    void binarize_int(LweSample* result, const LweSample* a, const int bit_size, TFheGateBootstrappingCloudKeySet* bk);
    void binarize(tBit* result, const tMultiBit* a, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);

    // convert binary numbers to integers (1 -> 1 mod 2^x, 0 -> -1 mod 2^x)
    void unbinarize_int(LweSample* result, const LweSample* a, TFheGateBootstrappingCloudKeySet* bk);

    void shift(tMultiBit* result, tMultiBit* in1, uint8_t input_bits, uint8_t shift_bits, TFheGateBootstrappingCloudKeySet* bk) ;
    void relu(tFixedPoint* result, tMultiBit* in1, uint8_t input_bits, TFheGateBootstrappingCloudKeySet* bk) ;

    // read ptxt weights
    void get_filters(FILE* fd_in, tBit* p_filt_b, uint32_t len, TFheGateBootstrappingCloudKeySet* bk);
    void get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_tern, uint32_t len, float thresh, TFheGateBootstrappingCloudKeySet* bk) ;
    void get_intfilters(FILE* fd_in, tMultiBit* p_filt_b, uint32_t len, TFheGateBootstrappingCloudKeySet* bk) ;
    void get_intfilters_ptxt(FILE* fd_in, uint32_t* p_filt_mb, uint32_t len);

    // helper function
    int pow_int(int base, int exponent);

}

#endif

#ifndef _BIN_OPS_GPU_H_
#define _BIN_OPS_GPU_H_

#include <cstdio>
#include "Layer.cuh"
#include "gates.cuh"
namespace BinOps
{
    // XNOR (1 bootstrap)
    void multiply(tBit* result, const tBit* a, const uint8_t b, redcufhe::Stream curr_sm);

    void multiply_pc_ints(redcufhe::Ctxt& result, redcufhe::Ctxt& in1, const uint16_t* multicand, uint8_t in1_bits, uint8_t in2_bits, redcufhe::Stream curr_sm);

    void add_bit(tMultiBit* result, const tBit* a, const tBit* b, redcufhe::Stream curr_sm);

    // Unsigned Adder (3 * MAX(a->size,b->size) - 1 bootstraps)
    void add(tMultiBit* result, const tMultiBit* a, const tMultiBit* b, uint8_t bits, redcufhe::Stream curr_sm);

    void add_pc_ints(redcufhe::Ctxt& result, redcufhe::Ctxt& in1, const uint16_t* addend, uint8_t in_bits, redcufhe::Stream curr_sm);

    void int_add(redcufhe::Ctxt& result, const redcufhe::Ctxt& a, const redcufhe::Ctxt& b, redcufhe::Stream curr_sm);

    // Unsigned Incrementer (2 * a->size - 1 bootstraps)
    void inc(tMultiBit* result, const tMultiBit* a, const tBit* b, redcufhe::Stream curr_sm);

    // Compute MAX(bit1, bit2) == OR(bit1, bit2)
    void max(tBit* result, const tBit* a, const tBit* b, redcufhe::Stream curr_sm);

    void shift(tMultiBit* result, tMultiBit* in1, uint8_t input_bits, uint8_t shift_bits, redcufhe::Stream curr_sm);

    void relu(tFixedPoint* result, tMultiBit* in1, uint8_t input_bits, redcufhe::Stream curr_sm) ;

    // extract top bit of tMultiBit instance
    void binarize_int(redcufhe::Ctxt& result, redcufhe::Stream curr_sm);
    void binarize(tBit* result, const tMultiBit* a);

    void unbinarize_int(redcufhe::Ctxt& result, redcufhe::Stream curr_sm);
    void unbinarize_int_inv(redcufhe::Ctxt& result, redcufhe::Stream curr_sm);

    // read ptxt weights
    void get_filters(FILE* fd_in, tBitPacked** p_filt_b, uint32_t len);

    void get_bitfilters(FILE* fd_in, tBitPacked** p_filt_b, uint32_t len);
    void get_intfilters(FILE* fd_in, tMultiBitPacked** p_filt_b, uint32_t len);
    void get_intfilters_ptxt(FILE* fd_in, uint16_t* p_filt_mb, uint32_t len);
    void get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_tern, uint32_t len, float thresh);
}

#endif

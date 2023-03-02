#ifndef _INT_OPS_GPU_H_
#define _INT_OPS_GPU_H_

#include "Layer.cuh"
#include "gates.cuh"
namespace IntOps
{

    void add(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b, redcufhe::Stream curr_sm);

    void subtract(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b, redcufhe::Stream curr_sm);

    // extract sign bit of tMultiBit instance representing a signed number
    void binarize(tBit* result, const tFixedPoint* a, redcufhe::Stream curr_sm);

    void binarize_int(tBit* result, redcufhe::Stream curr_sm);

    void invert(tFixedPoint* result, const tFixedPoint* a, const uint8_t* b, redcufhe::Stream curr_sm);

    void multiply_pc_ints(redcufhe::Ctxt& result, redcufhe::Ctxt& in1, const uint16_t* multicand, uint8_t in1_bits, uint8_t in2_bits, redcufhe::Stream curr_sm);

    void add_pc_ints(redcufhe::Ctxt& result, redcufhe::Ctxt& in1, const uint16_t* addend, uint8_t in1_bits, redcufhe::Stream curr_sm);

    void relu(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, redcufhe::Stream curr_sm);

    void shift(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, uint8_t shift_bits, redcufhe::Stream curr_sm);

}
#endif

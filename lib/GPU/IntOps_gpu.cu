#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "Layer.cuh"
#include "IntOps_gpu.cuh"
#include "BinOps_gpu.cuh"
#include <chrono>
#include <cstdio>
#include <iostream>

using namespace std::chrono;
using namespace std;
using namespace redcufhe;

void IntOps::add(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b, Stream curr_sm) {
    add_int(result->ctxt[0], a->ctxt[0], b->ctxt[0], curr_sm);
}

void IntOps::subtract(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b, Stream curr_sm) {
    sub_int(result->ctxt[0], a->ctxt[0], b->ctxt[0], curr_sm);
}

void IntOps::binarize(tBit* result, const tFixedPoint* a, Stream curr_sm) {
    Copy(*result, a->ctxt[a->size - 1], curr_sm);
}

void IntOps::binarize_int(tBit* result, Stream curr_sm) {
    redsec_binarize_bootstrap(*result, curr_sm);
}

void IntOps::invert(tFixedPoint* result, const tFixedPoint* a, const uint8_t* b, Stream curr_sm) {
    result->size = a->size;
    result->ctxt = new Ctxt[result->size];
    for (int i = 0; i < result->size; i++)
    {
        if (*b == 1) {
          Copy(result->ctxt[i], &a->ctxt[i], curr_sm);
        }
        else {
          Not(result->ctxt[i], &a->ctxt[i], curr_sm);
        }
    }
}

void IntOps::multiply_pc_ints(Ctxt& result, Ctxt& in1, const uint16_t* multicand, uint8_t in1_bits, uint8_t in2_bits, redcufhe::Stream curr_sm) {
    mul_int(result, in1, *multicand);
}

void IntOps::add_pc_ints(Ctxt& result, Ctxt& in1, const uint16_t* addend, uint8_t in1_bits, redcufhe::Stream curr_sm) {
    // Encode plaintext context
    int32_t conv_int = 0;
    conv_int = conv_int + (int32_t)((*addend)&0xFF);
    Ctxt* enc_int = new Ctxt[1];
    levelCONSTANT(enc_int[0], conv_int);

    add_int(result, in1, enc_int[0], curr_sm);
    delete [] enc_int;
}

void IntOps::relu(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, redcufhe::Stream curr_sm) {
  // check if result is already instantiated
  if (result->size != input_bits) {
    result->size = input_bits;
    result->ctxt = new Ctxt[result->size];
  }

  //AND each bit with MSB (val.ctxt[in_bits-1])- only keep values greater than threshold
  //do not set top bit, recenter around 0
  for(uint8_t i = 0 ; i < input_bits-1; i++)
  {
    bootsAND(result->ctxt[i], in1->ctxt[i], in1->ctxt[input_bits-1], curr_sm);
  }
}

void IntOps::shift(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, uint8_t shift_bits, redcufhe::Stream curr_sm) {
      //shift over result to keep values small
    for (int i = 0; i < input_bits; i++) {
      if ((i+shift_bits) > (input_bits-1)) { // sign extend
        Copy(result->ctxt[i], in1->ctxt[input_bits-1]);
      }
      else {
        Copy(result->ctxt[i], in1->ctxt[i+shift_bits]);
      }
    }
}

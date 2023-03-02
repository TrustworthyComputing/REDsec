/*********************************************************************************
*    FILE:        IntOps.cpp
*    Desc:        Contains basic operations on encrypted integers
*    Authors:     Lars Folkerts, Charles Gouert
*    Notes:       None
*********************************************************************************/
#ifdef ENCRYPTED
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "Layer.h"
#include "IntOps_enc.h"
#include "BinOps_enc.h"
#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace std;

void IntOps::invert(tFixedPoint* result, const tFixedPoint* a, const uint8_t* b, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk)
{
  result->size = a->size;
  result->ctxt = new_gate_bootstrapping_ciphertext_array(result->size, bk->params);
  for (int i = 0; i < result->size; i++)
  {
    if (*b == 1) {
        bootsCOPY(&result->ctxt[i], &a->ctxt[i], bk);
    }
    else {
        bootsNOT(&result->ctxt[i], &a->ctxt[i], bk);
    }
  }
}

void IntOps::add(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b,
	uint8_t bits, TFheGateBootstrappingCloudKeySet* bk) {
  const LweParams *in_out_params = bk->params->in_out_params;
  lweClear(&(result->ctxt[0]), in_out_params);
  lweAddTo(&(result->ctxt[0]), &(a->ctxt[0]), in_out_params);
  lweAddTo(&(result->ctxt[0]), &(b->ctxt[0]), in_out_params);
}

void IntOps::add_inplace(tFixedPoint* result, const tFixedPoint* a,
  uint8_t bits, TFheGateBootstrappingCloudKeySet* bk) {
  const LweParams *in_out_params = bk->params->in_out_params;
  lweAddTo(&(result->ctxt[0]), &(a->ctxt[0]), in_out_params);
}

void IntOps::subtract(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b,
  uint8_t in1_bits, TFheGateBootstrappingCloudKeySet* bk) {
  const LweParams *in_out_params = bk->params->in_out_params;
  result->size = 1;
  result->ctxt = new_LweSample(in_out_params);
  lweCopy(result->ctxt, a->ctxt, in_out_params);
  lweSubTo(result->ctxt, b->ctxt, in_out_params);
}

void IntOps::relu(tFixedPoint* result, tFixedPoint* in1, uint8_t in_bits,  TFheGateBootstrappingCloudKeySet* bk)
{
  //do not set top bit, recenter around 0
  for(uint8_t i = 0 ; i < in_bits-1; i++)
  {
    bootsAND(&result->ctxt[i], &in1->ctxt[i], &in1->ctxt[in_bits-1], bk);
  }
}

void IntOps::shift(tFixedPoint* result, tFixedPoint* in1, uint8_t in_bits, uint8_t shift_bits, TFheGateBootstrappingCloudKeySet* bk)
{
  if (result->size != in_bits) {
    result->size = in_bits;
    result->ctxt = new_gate_bootstrapping_ciphertext_array(result->size, bk->params);
  }

  assert((in_bits>0) && (shift_bits <= in_bits));
  //shift over result to keep values small
  for (int i = 0; i < in_bits; i++) {
    if ((i+shift_bits) > (in_bits-1)) { // sign extend
      bootsCOPY(&result->ctxt[i], &in1->ctxt[in_bits-1], bk);
    }
    else {
      bootsCOPY(&result->ctxt[i], &in1->ctxt[i+shift_bits], bk);
    }
  }
}

#endif //ENCRYPTED

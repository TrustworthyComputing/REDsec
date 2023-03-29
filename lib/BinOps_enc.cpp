/*********************************************************************************
*    FILE:       BinOps.c
*    Desc:       Definition of the BinOps (binary operations) namespace
*    Authors:    Lars Folkerts, Charles Gouert
*    Notes:      None
*********************************************************************************/
#ifdef ENCRYPTED
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cstring>
#include "BinOps_enc.h"
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
using namespace std;

void BinOps::multiply(tBit* result, const tBit* a, const uint8_t b, TFheGateBootstrappingCloudKeySet* bk)
{
  if (b == 0) {
    bootsNOT(&result[0], &a[0], bk);
  }
  else {
    bootsCOPY(&result[0], &a[0], bk);
  }
}

void BinOps::multiply_pc_ints(LweSample* result, LweSample* in1, const uint32_t* multicand, uint8_t in1_bits, uint8_t in2_bits, TFheGateBootstrappingCloudKeySet* bk) 
{
  const LweParams *in_out_params = bk->params->in_out_params;
  lweAddMulTo(result, *multicand,  in1, in_out_params);
}

void BinOps::add_bit(tMultiBit* result, const tBit* a, const tBit* b, TFheGateBootstrappingCloudKeySet* bk)
{
  result->ctxt = new_gate_bootstrapping_ciphertext_array(2, bk->params);
  result->size = 2;

  // compute sum
  bootsXOR(&result->ctxt[0], a, b, bk);

  // compute carry
  bootsAND(&result->ctxt[1], a, b, bk);
}

void BinOps::add(tMultiBit* result, const tMultiBit* a, const tMultiBit* b,
	uint8_t bits, TFheGateBootstrappingCloudKeySet* bk)
{
  uint32_t aligned_size = bits;
  if (!aligned_size) {
    aligned_size = (a->size > b->size) ? a->size : b->size;
  }

  if (aligned_size != result->size) {
    delete_gate_bootstrapping_ciphertext_array(result->size, result->ctxt);
    result->size = aligned_size;
    result->ctxt = new_gate_bootstrapping_ciphertext_array(aligned_size, bk->params);
  }

  tMultiBit carry;
  carry.size = aligned_size;
  carry.ctxt = new_gate_bootstrapping_ciphertext_array(carry.size, bk->params);

  tBit* temp = new_gate_bootstrapping_ciphertext_array(3, bk->params);

  tMultiBit a_aligned;
  a_aligned.size = aligned_size;
  a_aligned.ctxt = new_gate_bootstrapping_ciphertext_array(a_aligned.size, bk->params);

  tMultiBit b_aligned;
  b_aligned.size = aligned_size;
  b_aligned.ctxt = new_gate_bootstrapping_ciphertext_array(b_aligned.size, bk->params);

  for (int i = 0; i < aligned_size; i++) {
    if (i >= a->size) {
        bootsCONSTANT(&a_aligned.ctxt[i], 0, bk);
    }
    else {
        bootsCOPY(&a_aligned.ctxt[i], &a->ctxt[i], bk);
    }

    if (i >= b->size) {
        bootsCONSTANT(&b_aligned.ctxt[i], 0, bk);
    }
    else {
        bootsCOPY(&b_aligned.ctxt[i], &b->ctxt[i], bk);
    }
  }
  //initialize first carry to 0
  bootsCOPY(&carry.ctxt[0], 0, bk);

  //run full adders
  for (int i = 0; i < (aligned_size - 1); i++) {
    // Compute sum
    bootsXOR(&temp[0], &a_aligned.ctxt[i], &b_aligned.ctxt[i], bk);
    bootsXOR(&result->ctxt[i], &carry.ctxt[i], &temp[0], bk);
    // Compute carry
    bootsAND(&temp[1], &carry.ctxt[i], &temp[0], bk);
    bootsAND(&temp[2], &a_aligned.ctxt[i], &b_aligned.ctxt[i], bk);
    bootsOR(&carry.ctxt[i+1], &temp[1], &temp[2], bk);
  }

  bootsXOR(&temp[0], &a_aligned.ctxt[aligned_size-1], &b_aligned.ctxt[aligned_size-1], bk);
  bootsXOR(&result->ctxt[aligned_size-1], &carry.ctxt[aligned_size-1], &temp[0], bk);

  delete_gate_bootstrapping_ciphertext_array(carry.size, carry.ctxt);
  delete_gate_bootstrapping_ciphertext_array(a_aligned.size, a_aligned.ctxt);
  delete_gate_bootstrapping_ciphertext_array(b_aligned.size, b_aligned.ctxt);
  delete_gate_bootstrapping_ciphertext_array(3, temp);
}

void BinOps::add_int(LweSample* result, const LweSample* a, const LweSample* b, TFheGateBootstrappingCloudKeySet* bk) {
  const LweParams *in_out_params = bk->params->in_out_params;
  lweClear(result, in_out_params);
  lweAddTo(result, a, in_out_params);
  lweAddTo(result, b, in_out_params);
}

void BinOps::add_int_inplace(LweSample* result, const LweSample* a,
                             TFheGateBootstrappingCloudKeySet* bk) {
  const LweParams *in_out_params = bk->params->in_out_params;
  lweAddTo(result, a, in_out_params);
}

void BinOps::add_pc_ints(LweSample* result, LweSample* in1, const uint16_t* addend, uint8_t in_bits, TFheGateBootstrappingCloudKeySet* bk) {
  int32_t conv_int = 0;
  conv_int = conv_int + (int32_t)((*addend)&0xFFFF);
  LweSample* enc_int = new_gate_bootstrapping_ciphertext_array(1, bk->params);
  Torus32 mu = modSwitchToTorus32(conv_int, MULTIBIT_SPACE);
  lweNoiselessTrivial(&enc_int[0], mu, bk->params->in_out_params);
  lweAddTo(result, in1, bk->params->in_out_params);
  lweAddTo(result, enc_int, bk->params->in_out_params);
  delete_gate_bootstrapping_ciphertext_array(1, enc_int);
}

void BinOps::inc(tMultiBit* result, const tMultiBit* a, const tBit* b,
	uint8_t bits, TFheGateBootstrappingCloudKeySet* bk) {
  tMultiBit carry;
  carry.size = a->size;
  carry.ctxt = new_gate_bootstrapping_ciphertext_array(carry.size, bk->params);
  result->size = a->size;
  result->ctxt = new_gate_bootstrapping_ciphertext_array(result->size, bk->params);

  bootsCOPY(&carry.ctxt[0], b, bk);

  for (int i = 0; i < (a->size - 1); i++) {
    bootsXOR(&result->ctxt[i], &carry.ctxt[i], &a->ctxt[i], bk);
    bootsAND(&carry.ctxt[i+1], &carry.ctxt[i], &a->ctxt[i], bk);
  }

  bootsXOR(&result->ctxt[result->size-1], &carry.ctxt[a->size-1], &a->ctxt[a->size-1], bk);
  delete_gate_bootstrapping_ciphertext_array(carry.size, carry.ctxt);
}

void BinOps::max(tBit* result, const tBit* a, const tBit* b, TFheGateBootstrappingCloudKeySet* bk)
{
  bootsOR(result, a, b, bk);
}

int BinOps::pow_int(int base, int exponent) {
    if (exponent == 0)
        return 1;

    int result = pow_int(base, exponent / 2);
    result *= result;

    if (exponent & 1)
            result *= base;

    return result;
}  

void BinOps::binarize_int(LweSample* result, const LweSample* a, const int bit_size, TFheGateBootstrappingCloudKeySet* bk)
{
  int msg_space;
  if (bit_size == 1) {
    msg_space = 8; // binary
  }
  else {
    msg_space = BinOps::pow_int(2, bit_size);
  }
  const Torus32 mu_boot = modSwitchToTorus32(1, msg_space);
  tfhe_bootstrap_FFT(result, bk->bkFFT, mu_boot, a);
}

void BinOps::unbinarize_int(LweSample* result, const LweSample* a, TFheGateBootstrappingCloudKeySet* bk)
{
  const Torus32 mu_boot = modSwitchToTorus32(1, MULTIBIT_SPACE);
  tfhe_bootstrap_FFT(result, bk->bkFFT, mu_boot, a);
}

void BinOps::binarize(tBit* result, const tMultiBit* a, uint8_t bits,
	TFheGateBootstrappingCloudKeySet* bk)
{
  bootsCOPY(result, &a->ctxt[a->size - 1], bk);
}

void BinOps::relu(tFixedPoint* result, tMultiBit* in1, uint8_t in_bits,  TFheGateBootstrappingCloudKeySet* bk)
{
  //do not set top bit, recenter around 0
  for(uint8_t i = 0 ; i < in_bits-1; i++)
  {
    bootsAND(&result->ctxt[i], &in1->ctxt[i], &in1->ctxt[in_bits-1], bk);
  }
}

void BinOps::shift(tMultiBit* result, tMultiBit* in1, uint8_t in_bits, uint8_t shift_bits, TFheGateBootstrappingCloudKeySet* bk)
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

void BinOps::get_filters(FILE* fd_in, tBit* p_filt_b, uint32_t len, TFheGateBootstrappingCloudKeySet* bk)
{
  p_filt_b = new_gate_bootstrapping_ciphertext_array(len, bk->params);
  float* p_filt_f = (float*) calloc(len, sizeof(float)) ;
  uint8_t* p_filt_u = (uint8_t*) calloc(len, sizeof(uint8_t)) ;

  fread((void*)(p_filt_f), sizeof(float), len, fd_in) ;
  for(uint32_t i = 0 ; i<len ; i++)
  {
    p_filt_u[i] = ((((float*)(p_filt_f))[i]) < 0)?0:1 ;
  }
  for(uint32_t i = 0 ; i<len ; i++)
  {
    bootsCONSTANT(&p_filt_b[i], p_filt_u[i], bk);
  }
  free(p_filt_f) ;
  free(p_filt_u) ;
}

void BinOps::get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, float thresh, TFheGateBootstrappingCloudKeySet* bk)
{
  //check data format
  uint8_t version = NULL_FMT ;
  size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in) ;
  assert(version == BIN_FMT || version == TERN_FMT) ;
  int NBITS = (version==BIN_FMT)?1:2 ;
  bool b_tern = (version == TERN_FMT) && (p_filt_tern != NULL) ;

  static const uint16_t bits = sizeof(bitpack_t)*8 ;
  uint32_t adj_len = (len*NBITS+bits-1)/bits ;
  bitpack_t* p_filt_pack = new bitpack_t [adj_len] ;

  sread = fread((void*)(p_filt_pack), sizeof(bitpack_t), adj_len, fd_in) ;
  for(uint32_t i = 0 ; i<adj_len ; i++)
  { //unpack bits
    for(int32_t j = 0 ;  j <= bits-1 ; j+=NBITS)
    {
      if((i*bits+j)/NBITS >= len) { continue ; }
      p_filt_b[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-1)) & 0x1) ;
      if(b_tern){ p_filt_tern[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-2)) & 0x1) ; }
    }
  }
  if((version != TERN_FMT) && (p_filt_tern != NULL)){ memset(p_filt_tern, 0, len) ; }
  delete p_filt_pack  ;
}

void BinOps::get_intfilters(FILE* fd_in, tMultiBit* p_filt_mb, uint32_t len, TFheGateBootstrappingCloudKeySet* bk)
{
  //check data format
  const LweParams *in_out_params = bk->params->in_out_params;
  uint8_t version = NULL_FMT ;
  size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in) ;
  assert((version == UINT32_FMT) || (version==INT32_FMT)) ;

  int32_t* int_filt = (int32_t*) calloc(len, sizeof(int32_t)) ;
  Torus32 mu;

  for (uint32_t i = 0; i < len; i++)
  {
    p_filt_mb[i].size = 1;
    p_filt_mb[i].ctxt = new_LweSample(in_out_params);
  }
  size_t size = fread((int32_t*)(int_filt), sizeof(int32_t), len, fd_in) ;

  for (uint32_t i = 0; i < len; i++) {
    mu = modSwitchToTorus32(int_filt[i], MULTIBIT_SPACE);
    lweNoiselessTrivial(&p_filt_mb[i].ctxt[0], mu, in_out_params);
  }
  free(int_filt) ;
}

void BinOps::get_intfilters_ptxt(FILE* fd_in, uint32_t* p_filt_mb, uint32_t len)
{
  uint8_t version = NULL_FMT;
  size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in);
  assert((version == UINT32_FMT) || (version == INT32_FMT));
  sread = fread((uint32_t*)(p_filt_mb), sizeof(uint32_t), len, fd_in) ;
}

#endif

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <cstring>
#include <omp.h>
#include "BinOps_gpu.cuh"

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
using namespace redcufhe;

void BinOps::multiply(tBit* result, const tBit* a, const uint8_t b, Stream curr_sm) {
    if (b == 0) {
      Not(*result, *a, curr_sm);
    }
    else {
      Copy(*result, *a, curr_sm);
    }
}

void BinOps::add_bit(tMultiBit* result, const tBit* a, const tBit* b, Stream curr_sm)
{
    result->ctxt = new Ctxt[2];
    result->size = 2;

    // compute sum
    bootsXOR(result->ctxt[0], *a, *b, curr_sm);

    // compute carry
    bootsAND(result->ctxt[1], *a, *b, curr_sm);
}

void BinOps::add(tMultiBit* result, const tMultiBit* a, const tMultiBit* b, uint8_t bits, Stream curr_sm) {

    uint8_t aligned_size = bits;
    result->size = aligned_size;
    result->ctxt = new Ctxt[result->size];

    tMultiBit carry;
    carry.size = aligned_size;
    carry.ctxt = new Ctxt[carry.size];

    tBit* temp = new Ctxt[3];

    tMultiBit a_aligned;
    a_aligned.size = aligned_size;
    a_aligned.ctxt = new Ctxt[a_aligned.size];

    tMultiBit b_aligned;
    b_aligned.size = aligned_size;
    b_aligned.ctxt = new Ctxt[b_aligned.size];

    for (int i = 0; i < aligned_size; i++) {
      if (i >= a->size)
          levelCONSTANT(a_aligned.ctxt[i], 0);
      else
          Copy(a_aligned.ctxt[i], a->ctxt[i], curr_sm);

      if (i >= b->size)
          levelCONSTANT(b_aligned.ctxt[i], 0);
      else
          Copy(b_aligned.ctxt[i], b->ctxt[i], curr_sm);
    }

    //initialize first carry to 0
    levelCONSTANT(carry.ctxt[0], 0);

    //run full adders
    for (int i = 0; i < (aligned_size - 1); i++) {
      bootstrapped_full_adder(result->ctxt[i], carry.ctxt[i+1], temp[0],
        temp[1], a_aligned.ctxt[i], b_aligned.ctxt[i],
                           carry.ctxt[i], curr_sm);
    }

    bootsXOR(temp[0], a_aligned.ctxt[aligned_size-1], b_aligned.ctxt[aligned_size-1], curr_sm);
    bootsXOR(result->ctxt[aligned_size-1], carry.ctxt[aligned_size-1], temp[0], curr_sm);

    delete [] carry.ctxt;
    delete [] a_aligned.ctxt;
    delete [] b_aligned.ctxt;
    delete [] temp;
}

void BinOps::int_add(Ctxt& result, const Ctxt& a, const Ctxt& b, redcufhe::Stream curr_sm) {
    add_int(result, a, b, curr_sm);
}

void BinOps::multiply_pc_ints(Ctxt& result, Ctxt& in1, const uint16_t* multicand, uint8_t in1_bits, uint8_t in2_bits, redcufhe::Stream curr_sm)
{
    mul_int(result, in1, *multicand);
}

void BinOps::add_pc_ints(Ctxt& result, Ctxt& in1, const uint16_t* addend, uint8_t in_bits, redcufhe::Stream curr_sm)
{
    // Encode plaintext context
    int32_t conv_int = 0;
    conv_int = conv_int + (int32_t)((*addend)&0xFFFF);
    Ctxt* enc_int = new Ctxt[1];
    levelCONSTANT(enc_int[0], conv_int);

    add_int(result, in1, enc_int[0], curr_sm);
    delete [] enc_int;
}

void BinOps::inc(tMultiBit* result, const tMultiBit* a, const tBit* b, Stream curr_sm) {
    tMultiBit carry;
    carry.size = a->size;
    carry.ctxt = new Ctxt[carry.size];
    result->size = a->size;
    result->ctxt = new Ctxt[result->size];

    Copy(carry.ctxt[0], b);

    for (int i = 0; i < (a->size - 1); i++) {
        bootsXOR(result->ctxt[i], carry.ctxt[i], a->ctxt[i], curr_sm);
        bootsAND(carry.ctxt[i+1], carry.ctxt[i], a->ctxt[i], curr_sm);
    }
    bootsXOR(result->ctxt[result->size-1], carry.ctxt[a->size-1], a->ctxt[a->size-1], curr_sm);
    delete [] carry.ctxt;
}

void BinOps::max(tBit* result, const tBit* a, const tBit* b, Stream curr_sm) {
    bootsOR(*result, *a, *b, curr_sm);
}

void BinOps::binarize(tBit* result, const tMultiBit* a) {
    Copy(*result, a->ctxt[a->size - 1]);
}

void BinOps::binarize_int(Ctxt& result, redcufhe::Stream curr_sm) {
    redsec_binarize_bootstrap(result, curr_sm);
}

void BinOps::unbinarize_int_inv(Ctxt& result, redcufhe::Stream curr_sm) {
    redsec_unbinarize_bootstrap_inv(result, curr_sm);
}

void BinOps::unbinarize_int(Ctxt& result, redcufhe::Stream curr_sm) {
    redsec_unbinarize_bootstrap(result, curr_sm);
}

void BinOps::relu(tFixedPoint* result, tMultiBit* in1, uint8_t in_bits, redcufhe::Stream curr_sm)
{
    // check if result is already instantiated
    if (result->size != in_bits) {
      result->size = in_bits;
      result->ctxt = new Ctxt[result->size];
    }

  	//AND each bit with MSB (val.ctxt[in_bits-1])- only keep values greater than threshold
    //do not set top bit, recenter around 0
    for(uint8_t i = 0 ; i < in_bits-1; i++)
    {
      bootsAND(result->ctxt[i], in1->ctxt[i], in1->ctxt[in_bits-1], curr_sm);
    }
}

void BinOps::shift(tMultiBit* result, tMultiBit* in1, uint8_t in_bits, uint8_t shift_bits, redcufhe::Stream curr_sm)
{
    if (result->size != in_bits) {
      result->size = in_bits;
      result->ctxt = new Ctxt[result->size];
    }

    assert((in_bits>0) && (shift_bits <= in_bits));
    	//shift over result to keep values small
    for (int i = 0; i < in_bits; i++) {
      if ((i+shift_bits) > (in_bits-1)) { // sign extend
        Copy(result->ctxt[i], in1->ctxt[in_bits-1]);
      }
      else {
        Copy(result->ctxt[i], in1->ctxt[i+shift_bits]);
      }
    }
}

void BinOps::get_filters(FILE* fd_in, tBitPacked** p_filt_b, uint32_t len)
{
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(p_filt_b)
    for (int i = 0; i < NUM_GPUS; i++) {
      cudaSetDevice(i);
      (*p_filt_b)->enc_segs[i] = new redcufhe::Ctxt[len];
    }
    (*p_filt_b)->size = (uint8_t) len;
    float* p_filt_f = (float*) arr_calloc(len, sizeof(float)) ;
    uint8_t* p_filt_u = (uint8_t*) arr_calloc(len, sizeof(uint8_t)) ;

    fread((void*)(p_filt_f), sizeof(float), len, fd_in) ;
    for(uint32_t i = 0 ; i<len ; i++)
    {
        p_filt_u[i] = ((((float*)(p_filt_f))[i]) < 0)?0:1 ;
    }
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(p_filt_b)
    for (int i = 0; i < NUM_GPUS; i++) {
      for (uint32_t j = 0; j < len; j++) {
        cudaSetDevice(i);
        levelCONSTANT((*p_filt_b)->enc_segs[i][j], p_filt_u[j]);
      }
    }
    free(p_filt_f) ;
    free(p_filt_u) ;
}

void BinOps::get_ternfilters(FILE* fd_in, uint8_t* p_filt_b, uint8_t* p_filt_tern, uint32_t len, float thresh)
{
    //check data format
    uint8_t version = NULL_FMT ;
    size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in) ;
    assert(version == BIN_FMT || (version == TERN_FMT)) ;
    int NBITS = (version==BIN_FMT)?1:2 ;
    bool b_tern = (version == TERN_FMT) && (p_filt_tern != NULL) ;

    static const uint16_t bits = sizeof(bitpack_t)*8 ;
    uint32_t adj_len = (len*NBITS+bits-1)/bits ;
    bitpack_t* p_filt_pack = new bitpack_t [adj_len] ;

    sread = fread((void*)(p_filt_pack), sizeof(bitpack_t), adj_len, fd_in) ;

    for(uint32_t i = 0 ; i<adj_len ; i++)
    {  //unpack bits
    	for(int32_t j = 0 ;  j <= bits-1 ; j+=NBITS)
    	{
    		if((i*bits+j)/NBITS >= len) { continue ; }
        p_filt_b[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-1)) & 0x1) ;
        if(b_tern){ p_filt_tern[(i*bits+j)/NBITS] = ((p_filt_pack[i]>>(bits-j-2)) & 0x1) ; }
    	}
    }
    if((version != TERN_FMT) && (p_filt_tern != NULL)){ memset(p_filt_tern, 0, len) ; }
    delete p_filt_pack ;
}

void BinOps::get_bitfilters(FILE* fd_in, tBitPacked** p_filt_b, uint32_t len)
{
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(p_filt_b)
    for (int i = 0; i < NUM_GPUS; i++) {
      cudaSetDevice(i);
      (*p_filt_b)->enc_segs[i] = new redcufhe::Ctxt[len];
    }
    (*p_filt_b)->size = (uint8_t) len;

    uint16_t bits = sizeof(bitpack_t)*8 ;
    uint32_t adj_len = (len+bits-1)/bits ;
    bitpack_t* p_filt_pack = (bitpack_t*) arr_calloc(adj_len, sizeof(bitpack_t)) ;

    size_t sread = fread((void*)(p_filt_pack), sizeof(bitpack_t), adj_len, fd_in) ;

    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(p_filt_b)
    for (int k = 0; k < NUM_GPUS; k++) {
      for(uint32_t i = 0 ; i<adj_len ; i++)
      {  //unpack bits
      	for(int32_t j = bits-1 ;  j >= 0 ; j--)
      	{
  		if(i*bits+j >= len) { continue ; }
      		levelCONSTANT((*p_filt_b)->enc_segs[k][i*bits+j],(p_filt_pack[i]>>(bits-j-1)) & 0x1) ;
      	}
      }
    }
    free(p_filt_pack) ;
}

void BinOps::get_intfilters(FILE* fd_in, tMultiBitPacked** p_filt_mb, uint32_t len)
{
    uint8_t version = NULL_FMT;
    size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in);
    assert((version == UINT32_FMT) || (version == INT32_FMT));

    int32_t* int_filt = (int32_t*) calloc(len, sizeof(int32_t));

    size_t size = fread((int32_t*)(int_filt), sizeof(int32_t), len, fd_in) ;

    mbit_calloc_global(p_filt_mb, len, 1);
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(p_filt_mb)
    for (int k = 0; k < NUM_GPUS; k++) {
      for (uint32_t i = 0; i < len; i++) {
        Torus mu = ModSwitchToTorus(int_filt[i], MSG_SPACE);
        NoiselessTrivial((*p_filt_mb)->enc_segs[k][i].ctxt[0], mu);
      }
    }
    free(int_filt);
}

void BinOps::get_intfilters_ptxt(FILE* fd_in, uint16_t* p_filt_mb, uint32_t len)
{
    uint8_t version = NULL_FMT;
    size_t sread = fread(&version, sizeof(uint8_t), 1, fd_in);
    assert((version == UINT32_FMT) || (version == INT32_FMT));

    int32_t* int_filt = (int32_t*) calloc(len, sizeof(int32_t));

    size_t size = fread((int32_t*)(int_filt), sizeof(int32_t), len, fd_in) ;

    p_filt_mb = (uint16_t*) calloc(len, sizeof(uint16_t));

    for (uint32_t i = 0; i < len; i++) {
      p_filt_mb[i] = int_filt[i] & 0xFFFF;
    }
    free(int_filt);
}

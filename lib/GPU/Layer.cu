#include<cstdint>
#include<cstdlib>
#include<cstdio>
#include "Layer.cuh"

void print_status(const char* s)
{
#ifdef _PRINT_STATUS_
    printf("%s", s) ;
#endif
}

uint64_t get_size(tRectangle* ws, uint16_t in_dep, uint16_t out_dep)
{
    return (ws->h) * (ws->w) * (in_dep) * (out_dep)  ;
}

void netParamsCpy(tNetParams* dest, tNetParams* src)
{
    memcpy(&(dest->conv), &(src->conv), sizeof(tConvParams)) ;
    memcpy(&(dest->pool), &(src->pool), sizeof(tPoolParams)) ;
    memcpy(&(dest->bnorm), &(src->bnorm), sizeof(tBNormParams)) ;
    dest->e_bias = src->e_bias ;
    dest->version = src->version ;
}

void* arr_calloc(uint32_t len, uint8_t type_size)
{
    return calloc(len, type_size) ;
}

void bit_calloc(tBit** ret, uint32_t len)
{
    *ret = new redcufhe::Ctxt[len];
    return;
}

void bit_calloc_global(tBitPacked** ret, uint32_t len)
{
    omp_set_num_threads(NUM_GPUS);
    *ret = new tBitPacked;
    #pragma omp parallel for shared(ret)
    for (int i = 0; i < NUM_GPUS; i++) {
      cudaSetDevice(i);
      (*ret)->enc_segs[i] = new redcufhe::Ctxt[len];
    }
    (*ret)->size = (uint8_t) len;
    return;
}

void mbit_calloc(tMultiBit** ret, uint32_t len, uint8_t bits)
{
    *ret = new tMultiBit[len];
    for (uint32_t i = 0; i < len; i++)
    {
      (*ret)[i].size = bits;
      (*ret)[i].ctxt = new redcufhe::Ctxt[(*ret)[i].size];
    }
    return;
}

void mbit_calloc_global(tMultiBitPacked** ret, uint32_t len, uint8_t bits)
{
    omp_set_num_threads(NUM_GPUS);
    *ret = new tMultiBitPacked;
    (*ret)->size = (uint8_t) len;
    #pragma omp parallel for shared(ret)
    for (int i = 0; i < NUM_GPUS; i++) {
      cudaSetDevice(i);
      (*ret)->enc_segs[i] = new tMultiBit[len];
      for (uint32_t j = 0; j < len; j++) {
        (*ret)->enc_segs[i][j].size = bits;
        (*ret)->enc_segs[i][j].ctxt = new redcufhe::Ctxt[bits];
      }
    }
    return;
}

void fixpt_calloc(tFixedPoint** ret, uint32_t len, uint8_t bits)
{
    *ret = new tFixedPoint[len];
    for (uint32_t i = 0; i < len; i++)
    {
      (*ret)[i].size = 1;
      (*ret)[i].ctxt = new redcufhe::Ctxt[1];
    }
    return;
}

void fixpt_calloc_global(tFixedPointPacked** ret, uint32_t len, uint8_t bits)
{
    omp_set_num_threads(NUM_GPUS);
    *ret = new tFixedPointPacked;
    (*ret)->size = (uint8_t) len;
    #pragma omp parallel for shared(ret)
    for (int i = 0; i < NUM_GPUS; i++) {
      cudaSetDevice(i);
      (*ret)->enc_segs[i] = new tFixedPoint[len];
      for (uint32_t j = 0; j < len; j++) {
        (*ret)->enc_segs[i][j].size = bits;
        (*ret)->enc_segs[i][j].ctxt = new redcufhe::Ctxt[bits];
      }
    }
    return;
}

void bit_free(uint32_t len, tBit* to_free)
{
    delete [] to_free;
}

void bit_free_global(tBitPacked* to_free) {
  omp_set_num_threads(NUM_GPUS);
  #pragma omp parallel for shared(to_free)
  for (int i = 0; i < NUM_GPUS; i++) {
    cudaSetDevice(i);
    delete [] to_free->enc_segs[i];
  }
  delete to_free;
}

void mbit_free(uint32_t len, tMultiBit* to_free)
{
    for (uint32_t i = 0; i < len; i++)
    {
      delete [] to_free[i].ctxt;
    }
    delete to_free;
}

void mbit_free_global(uint32_t len, tMultiBitPacked* to_free) {
  omp_set_num_threads(NUM_GPUS);
  #pragma omp parallel for shared(to_free)
  for (int i = 0; i < NUM_GPUS; i++) {
    cudaSetDevice(i);
    for (uint32_t j = 0; j < len; j++) {
      delete [] to_free->enc_segs[i][j].ctxt;
    }
    delete to_free->enc_segs[i];
  }
  delete to_free;
}

void fixpt_free(uint32_t len, tFixedPoint* to_free)
{
    for (uint32_t i = 0; i < len; i++)
    {
      delete [] to_free[i].ctxt;
    }
    delete to_free;
}

void fixpt_free_global(uint32_t len, tMultiBitPacked* to_free) {
  omp_set_num_threads(NUM_GPUS);
  #pragma omp parallel for shared(to_free)
  for (int i = 0; i < NUM_GPUS; i++) {
    cudaSetDevice(i);
    for (uint32_t j = 0; j < len; j++) {
      delete [] to_free->enc_segs[i][j].ctxt;
    }
    delete to_free->enc_segs[i];
  }
  delete to_free;
}

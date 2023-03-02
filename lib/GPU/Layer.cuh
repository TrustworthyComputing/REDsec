#ifndef _LAY_H_
#define _LAY_H_

#include "REDcuFHE/redcufhe_gpu.cuh"
#include "gates.cuh"
#include <omp.h>
typedef redcufhe::PubKey TFheGateBootstrappingCloudKeySet ;

/********************************** Configurations *********************************/
#define _PRINT_STATUS_  //mostly used for Encrypted debugging
#define _PARALLEL_COMPUTE_
/********************************** Constants ***********************************/
#define SIZE_EMPTY 1    //Null dimension (arr length 1)
#define SINGLE_BIT 1
#define NUM_GPUS 1
#define MULTIBIT_BITS 26
#define FIXEDPOINT_BITS 26
#define MSG_SPACE 512
/********************************** typedefs ***********************************/
typedef float tFloat ;          //floating point arithmetic
typedef redcufhe::Ctxt tBit;
typedef struct tBitPackeds //arrays of encrypted bits
{
  tBit* enc_segs[NUM_GPUS];
  uint8_t size;
} tBitPacked;
typedef struct tMultiBits // array of encrypted integers (1 GPU)
{
  tBit* ctxt ;
  uint32_t size;
  uint8_t gpu_id;
} tMultiBit;
typedef struct tMultiBitPackeds // arrays of encrypted ints split across all GPUs
{
  tMultiBit* enc_segs[NUM_GPUS];
  uint8_t size;
} tMultiBitPacked;
typedef tMultiBit tFixedPoint;
typedef tMultiBitPacked tFixedPointPacked;

/*********************************** Enums ***********************************/
typedef enum _CONVTYPE
{
    E_NO_CONV,
    E_CONV,
    E_FC,
    E_FC_FINAL,
    NUM_CONVS,
}eConvType ;

typedef enum _POOLTYPE
{
    E_NO_POOL,
    E_MAXPOOL,
    E_SUMPOOL,
    NUM_POOLS,
}ePoolType ;

typedef enum _BIASTYPE
{
    E_NO_BIAS,
    E_BIAS,
    E_BNORM,
    NUM_BIASES,
}eBiasType ;

typedef enum _QUANT_TYPE
{
    E_ACTIVATION_NONE,
    E_ACTIVATION_SIGN,
    E_ACTIVATION_RELU,
    NUM_ACTIVATIONS
} eQuantType ;

typedef enum _ACTION
{
    E_INIT,
    E_PREP,
    E_EXEC,
    E_PREP_BIAS,
    E_EXPORT,
    NUM_ACTIONS,
}eAction ;

/*********************************** Structs ***********************************/
typedef struct _WDSZ
{
    int16_t h ; //height
    int16_t w ; //width
} tRectangle ;

//Dimensions
typedef struct _DIMS
{
	//dimensions
    tRectangle hw  ;
    uint32_t in_dep ;
    //bit sizes
    uint8_t in_bits ;
    uint8_t out_bits ;
    uint8_t filter_bits ;
    uint8_t bias_bits ;
    //max value
    uint32_t up_bound ;
    float scale ;
} tDimensions ;


typedef struct _CONV_PARAMS
{
   tRectangle window ;
   uint16_t w_max ;
   bool same_pad ;
   float tern_thresh ;
   tRectangle stride;
} tConvParams;

typedef struct _BNORM_PARAMS
{
    bool use_scale ;
    float eps ; //epsilon
} tBNormParams ;

typedef struct _POOL_PARAMS
{
   tRectangle window ;
   bool same_pad ;
   tRectangle stride;
} tPoolParams ;

typedef struct _QUANT_PARAMS
{
    uint8_t shift_bits ;
} tQParams ;

typedef struct _NET_PARAMS
{
    tConvParams conv ;
    tPoolParams pool ;
    tBNormParams bnorm ;
    tQParams quant ;
    eBiasType e_bias ;
    uint16_t version ;
} tNetParams ;

typedef union _ACT_PARAMS
{
    tDimensions* d ;
    tBitPacked* b;
    tFixedPointPacked* fp ; // array of encrypted ints
} tActParams ;

/**************************** Function Declarations *****************************/
uint64_t get_size(tRectangle* ws, uint16_t in_dep, uint16_t out_dep);
void netParamsCpy(tNetParams* dest, tNetParams* src) ;
void bit_calloc(tBit** ret, uint32_t len);
void bit_calloc_global(tBitPacked** ret, uint32_t len);
void* arr_calloc(uint32_t len, uint8_t type_size);
void mbit_calloc(tMultiBit** ret, uint32_t len, uint8_t bits);
void mbit_calloc_global(tMultiBitPacked** ret, uint32_t len, uint8_t bits);
void fixpt_calloc(tFixedPoint** ret, uint32_t len, uint8_t bits);
void fixpt_calloc_global(tFixedPointPacked** ret, uint32_t len, uint8_t bits);
void bit_free(uint32_t len, tBit* to_free);
void bit_free_global(tBitPacked* to_free);
void mbit_free(uint32_t len, tMultiBit* to_free);
void mbit_free_global(uint32_t len, tMultiBitPacked* to_free);
void fixpt_free(uint32_t len, tFixedPoint* to_free);
void fixpt_free_global(uint32_t len, tMultiBitPacked* to_free);

void print_status(const char* s);
#endif

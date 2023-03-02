/**************************************************************************************************
* FILENAME :        Layer.cpp
*               
* VERSION:          0.1
*
* DESCRIPTION :
*       General Purpose layer functions to copy and access data.
*
* NOTES :     
* AUTHOR :    Lars Folkerts
* START DATE :    16 Aug 20
*************************************************************************************************/

#ifndef _LAY_H_
#define _LAY_H_

#include<cstdint>
#ifdef ENCRYPTED
#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#else
typedef void TFheGateBootstrappingCloudKeySet ;
#endif
/********************************** Configurations *********************************/
#if defined(ENCRYPTED)
  #define _PRINT_STATUS_  //Encrypted debugging
  #define _PARALLEL_COMPUTE_
#else
#endif
/********************************** Constants ***********************************/
#define SIZE_EMPTY 1    //Null dimension (arr length 1)
#define SINGLE_BIT 1
#define MULTIBIT_BITS 12
#define FIXEDPOINT_BITS 12
#define MULTIBIT_SPACE 2048 // 2^MULTIBIT_BITS
/********************************** typedefs ***********************************/
typedef float tFloat ;          //floating point arithmatic
#ifdef ENCRYPTED
typedef LweSample tBit;
typedef struct tMultiBits
{
  tBit* ctxt ; //array of bits
  uint32_t size;
} tMultiBit;
typedef tMultiBit tFixedPoint;
#elif defined _WEIGHT_CONVERT_
typedef uint32_t tFixedPoint ;  //integer fixed point arithmatic
typedef float tMultiBit ;       //bitwise arithmatic
typedef uint8_t tBit ;          //bitwise arithmatic, resticted to 0,1
#else
typedef int32_t tFixedPoint ;   //integer fixed point arithmatic
typedef uint32_t tMultiBit ;    //bitwise arithmatic
typedef uint8_t tBit ;          //bitwise arithmatic, resticted to 0,1
#endif

/*********************************** Enums ***********************************/
//Convolution type
typedef enum _CONVTYPE
{
    E_NO_CONV,   //no convolution, for input layers
    E_CONV,      //standard 2D convolutional
    E_FC,        //fully connected layer
    E_FC_FINAL,  //final fully connected layer, no activtion to follow
    NUM_CONVS,
}eConvType ;

//pooling types
typedef enum _POOLTYPE
{
    E_NO_POOL,  //no pooling
    E_MAXPOOL,  //maxpooling, binary output layers only
    E_SUMPOOL,  //sumpooling
    NUM_POOLS,
}ePoolType ;

//bias types
typedef enum _BIASTYPE
{
    E_NO_BIAS,  //no bias, used for output layer
    E_BIAS,     //standard addition bias added to weights
    E_BNORM,    //Batch normalization
    NUM_BIASES,
}eBiasType ;

//activation
typedef enum _QUANT_TYPE
{
    E_ACTIVATION_NONE, //no activation, for output layers
    E_ACTIVATION_SIGN, //sign activation
    E_ACTIVATION_RELU, //relu activation
    NUM_ACTIVATIONS
} eQuantType ;

//Actions for Bin/IntLayer::run
typedef enum _ACTION
{
    E_INIT, 	  //initialize opjects
    E_PREP, 	  //prepare diemnsions
    E_EXEC, 	  //run neural network
    E_PREP_BIAS,  //weight convert - calculate batchnorm/bias term
    E_EXPORT,     //weight convert - export weights as compressed file
    NUM_ACTIONS,
}eAction ;

/*********************************** Structs ***********************************/
//Rectagle of height and width
typedef struct _WDSZ
{
    int16_t h ; //height
    int16_t w ; //width
} tRectangle ;

    //Dimensions
typedef struct _DIMS
{
	//dimensions
    tRectangle hw  ;       //height and width of image
    uint32_t in_dep ;      //input depth (neurons or channels)
    //bit sizes
    uint8_t in_bits ;      //input bits
    uint8_t out_bits ;     //output bits
    uint8_t filter_bits ;  //number of bits in filter
    uint8_t bias_bits ;    //number of bits for bias
    //max value
    uint32_t up_bound ;    //maximum possible value
    float scale ;          //scaling factor
} tDimensions ;


//Convolution paramerters
typedef struct _CONV_PARAMS
{
   tRectangle window ;    //convolution window
   bool same_pad ;        //same pad (TRUE) or valid pad (FALSE)
   float tern_thresh ;    //ternary weight threshold. i.e. if |w|<tern_thresh, w=0
   tRectangle stride ;    //stride height / width
} tConvParams;

//Batch Normalization parameters
typedef struct _BNORM_PARAMS
{
    bool use_scale ;      //use scaling (warning, untested)
    float eps ;           //epsilon 
} tBNormParams ;

//Pooling parameters
typedef struct _POOL_PARAMS
{
   tRectangle window ;   //pooling window 
   bool same_pad ;       //same pad (TRUE) or valid pad (FALSE)
   tRectangle stride ;   //stride height/width
} tPoolParams ;

//quantization parameters
typedef struct _QUANT_PARAMS
{
    uint8_t shift_bits ;   //amount of bits to shift in ReLU
} tQParams ;

//network parameters
typedef struct _NET_PARAMS
{
    tConvParams conv ;
    tPoolParams pool ;
    tBNormParams bnorm ;
    tQParams quant ;
    eBiasType e_bias ; 
    uint16_t version ;  //used to implement backwards compatibility
} tNetParams ;

//Action Paramters
typedef union _ACT_PARAMS
{
    tDimensions* d ;
    tBit* b;
    tFixedPoint* fp ; // array of encrypted ints
} tActParams ;

/**************************** Function Declarations *****************************/
uint64_t get_size(tRectangle* ws, uint16_t in_dep, uint16_t out_dep) ;
void netParamsCpy(tNetParams* dest, tNetParams* src) ;
tBit* bit_calloc(uint32_t len, TFheGateBootstrappingCloudKeySet* bk) ;
tMultiBit* mbit_calloc(uint32_t len, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk) ;
tFixedPoint* fixpt_calloc(uint32_t len, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk) ;
void bit_free(uint32_t len, tBit* to_free) ;
void mbit_free(uint32_t len, tMultiBit* to_free) ;
void fixpt_free(uint32_t len, tFixedPoint* to_free) ;
void print_status(const char* s) ;
#endif

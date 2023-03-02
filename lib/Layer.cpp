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

#include<cstdint>
#include<cstdlib>
#include<cstdio>
#include<cstring>
#include "Layer.h"

/*********************************************************************************
*    Func:      print_status
*    Desc:      Prints buffer depending on _PRINT_STATUS_ define
*    Inputs:    const char* - message to print
*    Return:    None.
*    Notes:     
*********************************************************************************/
void print_status(const char* s)
{
#ifdef _PRINT_STATUS_
    printf("%s", s) ;
#endif
}

/*********************************************************************************
*    Func:      get_size
*    Desc:      Gets size of weights (l*w*in*out)
*    Inputs:    tRectangle* - input height and width
*    		uint16_t - layer input depth
*		uint16_t - layer output depth
*    Return:    None.
*    Notes:
*********************************************************************************/
uint64_t get_size(tRectangle* ws, uint16_t in_dep, uint16_t out_dep)
{
    return (ws->h) * (ws->w) * (in_dep) * (out_dep)  ;
}

/*********************************************************************************
*    Func:      netParamsCpy
*    Desc:      Deep copy of a tNetParams object
*    Inputs:    tNetParams* - preallocated deep copy destination object
*    		tNetParams* - deep copy source object
*    Return:    None.
*    Notes:
*********************************************************************************/
void netParamsCpy(tNetParams* dest, tNetParams* src)
{
    memcpy(&(dest->conv), &(src->conv), sizeof(tConvParams)) ;
    memcpy(&(dest->pool), &(src->pool), sizeof(tPoolParams)) ;
    memcpy(&(dest->bnorm), &(src->bnorm), sizeof(tBNormParams)) ;
    dest->e_bias = src->e_bias ;
    dest->version = src->version ;
 
}

/*********************************************************************************
*    Func:      bit_calloc
*    Desc:      Allocates memory for tBit objects
*    Inputs:    uint32_t - number of tBit objects
*               TFheGateBootstrappingCloudKeySet* - bootstapping key 
*               	(or NULL for unencrypted)
*    Return:    None.
*    Notes:
*********************************************************************************/
tBit* bit_calloc(uint32_t len, TFheGateBootstrappingCloudKeySet* bk)
{
#ifdef ENCRYPTED
    return new_gate_bootstrapping_ciphertext_array(len, bk->params) ;
#else
    return (tBit*) calloc(len, sizeof(tBit)) ;
#endif
}

/*********************************************************************************
*    Func:      mbit_calloc
*    Desc:      Allocate memory for tMultiBit objects
*    Inputs:    uint32_t - number of tMultiBit objects
*    		uint8_t - number of bits in tMultiBit object
*               TFheGateBootstrappingCloudKeySet* - bootstapping key 
*                       (or NULL for unencrypted)
*    Return:    None.
*    Notes:
*********************************************************************************/
tMultiBit* mbit_calloc(uint32_t len, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk)
{
#ifdef ENCRYPTED
    tMultiBit* ret = new tMultiBit[len];
    for (uint32_t i = 0; i < len; i++)
    {
      ret[i].size = bits ;
      ret[i].ctxt = new_gate_bootstrapping_ciphertext_array(ret[i].size, bk->params);
    }
    return ret ;
#else
    return (tMultiBit*) calloc(len, sizeof(tMultiBit)) ;
#endif
}

/*********************************************************************************
*    Func:      fixpt_calloc
*    Desc:      Allocate memory for tFixedPoint objects
*    Inputs:    uint32_t - number of tFixedPoint objects
*               uint8_t - number of bits in tMultiBit object
*               TFheGateBootstrappingCloudKeySet* - bootstapping key
*                       (or NULL for unencrypted)
*    Return:    None.
*    Notes:
*********************************************************************************/
tFixedPoint* fixpt_calloc(uint32_t len, uint8_t bits, TFheGateBootstrappingCloudKeySet* bk)
{
#ifdef ENCRYPTED
    tFixedPoint* ret = new tFixedPoint[len];
    for (uint32_t i = 0; i < len; i++)
    {
      ret[i].size = bits ;
      ret[i].ctxt = new_gate_bootstrapping_ciphertext_array(ret[i].size, bk->params);
      for (uint32_t j = 0; j < ret[i].size; j++) {
          lweClear(&(ret[i].ctxt[j]), bk->params->in_out_params);
      }
    }
    return ret ;
#else
    return (tFixedPoint*) calloc(len, sizeof(tFixedPoint)) ;
#endif
}

/*********************************************************************************
*    Func:      bit_free
*    Desc:      Frees memory for tBit objects
*    Inputs:    uint32_t - number of tBit objects
*               tBit* - starting pointer of memory to free
*    Return:    None.
*    Notes:
*********************************************************************************/
void bit_free(uint32_t len, tBit* to_free)
{
#ifdef ENCRYPTED
    delete_gate_bootstrapping_ciphertext_array(len, to_free);
#else
    free(to_free) ;
#endif
}

/*********************************************************************************
*    Func:      mbit_free
*    Desc:      Frees memory for tMultiBit objects
*    Inputs:    uint32_t - number of tMultiBit objects
*               tBit* - starting pointer of memory to free
*    Return:    None.
*    Notes:
*********************************************************************************/
void mbit_free(uint32_t len, tMultiBit* to_free)
{
#ifdef ENCRYPTED
    for (uint32_t i = 0; i < len; i++)
    {
    	delete_gate_bootstrapping_ciphertext_array(to_free[i].size, to_free[i].ctxt);
    }
    delete to_free ;
#else
    free(to_free) ;
#endif
}

/*********************************************************************************
*    Func:      fixpt_free
*    Desc:      Frees memory for tFixedPoint objects
*    Inputs:    uint32_t - number of tFixedPoint objects
*               tBit* - starting pointer of memory to free
*    Return:    None.
*    Notes:
*********************************************************************************/
void fixpt_free(uint32_t len, tFixedPoint* to_free)
{
#ifdef ENCRYPTED
    for (uint32_t i = 0; i < len; i++)
    {
    	delete_gate_bootstrapping_ciphertext_array(to_free[i].size, to_free[i].ctxt);
    }
    delete to_free ;
#else
    free(to_free) ;
#endif
}

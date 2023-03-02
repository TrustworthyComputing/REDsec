#ifndef _INT_OPS_H_
#define _INT_OPS_H_

#include "Layer.h"
#ifdef ENCRYPTED
#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#endif
namespace IntOps
{
    // Signed "Multiplier" with +/- 1
    void multiply(tFixedPoint* result, const tFixedPoint* a, const tBit* b, 
    	uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);
    void invert(tFixedPoint* result, const tFixedPoint* a, const uint8_t* b,
    	 uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);

    // Signed Adder (3 * MAX(a->size,b->size) - 1 bootstraps)
    void add(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b, 
    	uint8_t in1_bits, TFheGateBootstrappingCloudKeySet* bk);

    void add_inplace(tFixedPoint* result, const tFixedPoint* a,
        uint8_t bits, TFheGateBootstrappingCloudKeySet* bk);

    void subtract(tFixedPoint* result, const tFixedPoint* a, const tFixedPoint* b,
        uint8_t in1_bits, TFheGateBootstrappingCloudKeySet* bk);
    
    void relu(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, 
        TFheGateBootstrappingCloudKeySet* bk) ;

    void shift(tFixedPoint* result, tFixedPoint* in1, uint8_t input_bits, 
        uint8_t shift_bits, TFheGateBootstrappingCloudKeySet* bk);
}
#endif

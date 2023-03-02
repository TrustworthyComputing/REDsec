#ifndef _GATES_H
#define _GATES_H

#include "REDcuFHE/redcufhe_gpu.cuh"
#include "REDcuFHE/redcufhe_bootstrap_gpu.cuh"
#include "REDcuFHE/redcufhe.h"
#include "REDcuFHE/ntt_gpu/ntt.cuh"
#include <vector>
#include "Layer.cuh"

void CtxtCopyD2H(const redcufhe::Ctxt& c, redcufhe::Stream st);
void CtxtCopyH2D(const redcufhe::Ctxt& c, redcufhe::Stream st);
void redsec_binarize_bootstrap(redcufhe::Ctxt& out, redcufhe::Stream st);
void redsec_unbinarize_bootstrap(redcufhe::Ctxt& out, redcufhe::Stream st);
void redsec_unbinarize_bootstrap_inv(redcufhe::Ctxt& out, redcufhe::Stream st);
void bootsNAND(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void bootsOR(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void bootsAND(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void bootsNOR(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void bootsXOR(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void bootsXNOR(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, const redcufhe::Ctxt& in1, redcufhe::Stream st);
void levelNOT(redcufhe::Ctxt& out, const redcufhe::Ctxt& in0, redcufhe::Stream st);
void NoiselessTrivial(redcufhe::Ctxt& result, redcufhe::Torus mu);
void levelCONSTANT(redcufhe::Ctxt& result, int32_t value);
void add_int(redcufhe::Ctxt& sum, const redcufhe::Ctxt& a, const redcufhe::Ctxt& b, redcufhe::Stream st);
void mul_int(redcufhe::Ctxt& prod, const redcufhe::Ctxt& a, uint16_t b);
void sub_int(redcufhe::Ctxt& res, const redcufhe::Ctxt& a, const redcufhe::Ctxt& b, redcufhe::Stream st);
void bootstrapped_full_adder(redcufhe::Ctxt& sum, redcufhe::Ctxt& carry_out, redcufhe::Ctxt& temp_a, redcufhe::Ctxt& temp_b, const redcufhe::Ctxt& a, const redcufhe::Ctxt& b, const redcufhe::Ctxt& carry_in, redcufhe::Stream st);
#endif

#include <iostream>
#include <ctime>
#include <chrono>
#include "gates.cuh"

using namespace std;
using namespace redcufhe;

void CtxtCopyH2D(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_device_->data(),
                  c.lwe_sample_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice,
                  st.st());
}

void CtxtCopyD2H(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_->data(),
                  c.lwe_sample_device_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost,
                  st.st());
}

__device__
void NoiselessTrivial(LWESample* result, Torus mu){
    const int32_t n = result->n();
    for (int32_t i = 0; i < n; ++i) result->a()[i] = 0;
    result->b() = mu;
}

__device__
void Copy_leveled(LWESample** out,
          const LWESample* in) {
  for (int i = 0; i <= in->n(); i ++)
    (*out)->data()[i] = in->data()[i];
}

__device__ inline
uint32_t ModSwitch2048(uint32_t a) {
  return (((uint64_t)a << 32) + (0x1UL << 52)) >> 53;
}

__global__
void NandOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 - in0[i] - in1[i];
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void OrOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 + in0[i] + in1[i];
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void AndOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 + in0[i] + in1[i];
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void NorOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 - in0[i] - in1[i];
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void XorOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 + 2*(in0[i] + in1[i]);
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void XnorOp(Torus* out, Torus* in0, Torus* in1, uint32_t n, const Torus fix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = 0 - 2*in0[i] - 2*in1[i];
  }
  if (i == n) {
    out[i] += fix;
  }
}

__global__
void NotOp(Torus* out, const Torus* in, uint32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = -in[i];
  }
}

void levelNOT(Ctxt& out,
          const Ctxt& in0,
          Stream st) {
  NotOp<<<1,512,0,st.st()>>>(out.lwe_sample_device_->data(), in0.lwe_sample_device_->data(), out.lwe_sample_device_->n());
}

void redsec_binarize_bootstrap(Ctxt& out, Stream st) {
  static const Torus mu = ModSwitchToTorus(-1,8);
  CtxtCopyH2D(out, st);
  Bootstrap(out.lwe_sample_device_, out.lwe_sample_device_, mu, st.st());
  CtxtCopyD2H(out, st);
  CuCheckError();
}

void redsec_unbinarize_bootstrap(Ctxt& out, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, MSG_SPACE);
  CtxtCopyH2D(out, st);
  Bootstrap(out.lwe_sample_device_, out.lwe_sample_device_, mu, st.st());
  CtxtCopyD2H(out, st);
}

void redsec_unbinarize_bootstrap_inv(Ctxt& out, Stream st) {
  static const Torus mu = ModSwitchToTorus(-1, MSG_SPACE);
  CtxtCopyH2D(out, st);
  Bootstrap(out.lwe_sample_device_, out.lwe_sample_device_, mu, st.st());
  CtxtCopyD2H(out, st);
}

void NoiselessTrivial(Ctxt& result, Torus mu){
    const int32_t n = result.lwe_sample_->n();

    for (int32_t i = 0; i < n; ++i) result.lwe_sample_->a()[i] = 0;
    result.lwe_sample_->b() = mu;
}

void levelCONSTANT(Ctxt& result, int32_t value) {
    static const Torus MU = ModSwitchToTorus(1, 8);
    NoiselessTrivial(result, value ? MU : -MU);
}

__global__
void AddOp(Torus* out, Torus* in0, Torus* in1, uint32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = in0[i] + in1[i];
  }
}

__global__
void SubOp(Torus* out, Torus* in0, Torus* in1, uint32_t n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (n+1)) {
    out[i] = in0[i] - in1[i];
  }
}

void mul_int(redcufhe::Ctxt& prod, const redcufhe::Ctxt& a, uint16_t b) {
  for (int i = 0; i < a.lwe_sample_->n(); i++) {
    prod.lwe_sample_->data()[i] = a.lwe_sample_->data()[i]*b;
  }
  prod.lwe_sample_->b() = a.lwe_sample_->b()*b;
  CuCheckError();
}

void add_int(Ctxt& sum, const Ctxt& a, const Ctxt& b, Stream st) {
    CtxtCopyH2D(a, st);
    CtxtCopyH2D(b, st);

    int numBlocks = (sum.lwe_sample_->n() + 512 - 1)/512;

    AddOp<<<numBlocks,512,0,st.st()>>>(sum.lwe_sample_device_->data(), a.lwe_sample_device_->data(),
        b.lwe_sample_device_->data(), sum.lwe_sample_device_->n());

    CtxtCopyD2H(sum, st);
    CuCheckError();
}

void sub_int(Ctxt& res, const Ctxt& a, const Ctxt& b, Stream st) {
    CtxtCopyH2D(a, st);
    CtxtCopyH2D(b, st);

    int numBlocks = (res.lwe_sample_->n() + 512 - 1)/512;

    SubOp<<<numBlocks,512,0,st.st()>>>(res.lwe_sample_device_->data(), a.lwe_sample_device_->data(),
        b.lwe_sample_device_->data(), res.lwe_sample_device_->n());

    CtxtCopyD2H(res, st);
    CuCheckError();
}

void bootstrapped_full_adder(Ctxt& sum, Ctxt& carry_out, Ctxt& temp_a, Ctxt& temp_b, const Ctxt& a, const Ctxt& b, const Ctxt& carry_in, Stream st) {
    // transfer to GPU
    CtxtCopyH2D(temp_a, st);
    CtxtCopyH2D(temp_b, st);
    CtxtCopyH2D(a, st);
    CtxtCopyH2D(b, st);
    CtxtCopyH2D(carry_in, st);

    // temp[0] = a XOR b
    static const Torus fix = ModSwitchToTorus(1, 4);
    static const Torus mu = ModSwitchToTorus(1, 8);
    XorBootstrap(temp_a.lwe_sample_device_, a.lwe_sample_device_,
        b.lwe_sample_device_, mu, fix, st.st());


    // sum = temp[0] XOR carry_in
    XorBootstrap(sum.lwe_sample_device_, temp_a.lwe_sample_device_,
        carry_in.lwe_sample_device_, mu, fix, st.st());

    // temp[0] = carry_in AND temp[0]
    static const Torus fix_and = ModSwitchToTorus(-1, 8);
    AndBootstrap(temp_a.lwe_sample_device_, carry_in.lwe_sample_device_,
        temp_a.lwe_sample_device_, mu, fix_and, st.st());

    // temp[1] = a AND b
    AndBootstrap(temp_b.lwe_sample_device_, a.lwe_sample_device_,
        b.lwe_sample_device_, mu, fix_and, st.st());

    // carry_out = temp[0] OR temp[1]
    static const Torus fix_or = ModSwitchToTorus(1, 8);
    OrBootstrap(carry_out.lwe_sample_device_, temp_a.lwe_sample_device_,
        temp_b.lwe_sample_device_, mu, fix_or, st.st());

    // transfer to CPU
    CtxtCopyD2H(sum, st);
    CtxtCopyD2H(carry_out, st);
}

void bootsNAND(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void bootsOR(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  OrBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void bootsAND(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  AndBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void bootsNOR(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  NorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void bootsXOR(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  XorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void bootsXNOR(Ctxt& out, const Ctxt& in0, const Ctxt& in1, Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  XnorBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_,
      in1.lwe_sample_device_, mu, fix, st.st());
}

void deviceCopy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  for (int i = 0; i <= in.lwe_sample_device_->n(); i ++)
    out.lwe_sample_device_->data()[i] = in.lwe_sample_device_->data()[i];
}

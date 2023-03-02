#include <cstdint>
#include <cstdio>
#include <assert.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <ratio>
#include <chrono>
#include <omp.h>
#include <iostream>
#include "IntFunc_gpu.cuh"
#include "Layer.cuh"

#include "IntOps_gpu.cuh"
#include "BinOps_gpu.cuh"

using namespace std;
using namespace std::chrono;
using namespace redcufhe;

IntFunc::Convolution::Convolution(uint16_t out_depth, tConvParams* in_params)
{
	//error checking
    assert(in_params != NULL) ;
    assert(out_depth > 0) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    OutDepth = out_depth ;
    memcpy(&conv, in_params, sizeof(conv)) ;
    b_prep = false ;
}

tDimensions* IntFunc::Convolution::prep(FILE* fd_filt, tDimensions* ret_dim)
{
    assert(!b_prep) ;
    assert((ret_dim != NULL) && (fd_filt != NULL)) ;
    assert((ret_dim->hw.h >= conv.window.h ) && (ret_dim->hw.w >= conv.window.w)) ;
    assert((conv.stride.h != 0) && (conv.stride.w != 0)) ;
    //set parameters depth
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;
    if(conv.same_pad)
    {   //zero padding -> offset is half of window size
        out_hw.h = (lay_dim.hw.h - 1)/conv.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/conv.stride.w + 1;
        if(conv.stride.h == 1){ offset_window.h = (int16_t)((conv.window.h-1)/2) ; }
	    else{ offset_window.h = (out_hw.h*conv.stride.h - lay_dim.hw.h)/2 ; }
        if(conv.stride.w == 1){ offset_window.w = (int16_t)((conv.window.w-1)/2) ; }
	    else{ offset_window.w = (out_hw.w*conv.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid padding - no offest, but remove left and right borders
    {
        offset_window.h = 0 ;
        offset_window.w = 0 ;
        out_hw.h = lay_dim.hw.h - 2*((int16_t)((conv.window.h-1)/2)) ;
        out_hw.h = (out_hw.h)/conv.stride.h  ;
        out_hw.w = lay_dim.hw.w - 2*((int16_t)((conv.window.w-1)/2)) ;
        out_hw.w = (out_hw.w)/conv.stride.w ;
    }
    //calculate number of additions for output bitsize
    in_up_bound = lay_dim.up_bound;
    lay_dim.up_bound *=
        (lay_dim.filter_bits) *
        (conv.window.w) *
        (conv.window.h) *
        (lay_dim.in_dep) ;

    for(lay_dim.out_bits = lay_dim.in_bits ;
        (lay_dim.up_bound >> lay_dim.out_bits) > 0 ;
        lay_dim.out_bits++) ;

    //allocate memory
    flen = get_size(&conv.window, lay_dim.in_dep, OutDepth) ;
    p_filters = (uint8_t*) calloc(flen, sizeof(uint8_t)) ;
    p_tern = (uint8_t*) calloc(flen, sizeof(uint8_t)) ;

    //get filters
    BinOps::get_ternfilters(fd_filt, p_filters, p_tern, flen, conv.tern_thresh) ;

    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_dep = OutDepth ;
    ret_dim->in_bits = lay_dim.out_bits  ;
    ret_dim->out_bits = SINGLE_BIT ; //clear last dimension
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale ;

    uint32_t len = get_size(&out_hw, SIZE_EMPTY, OutDepth) ;
    fixpt_calloc_global(&p_output, len, 1) ;
    b_prep = true ;

    return ret_dim ;
}

tFixedPointPacked* IntFunc::Convolution::execute(tFixedPointPacked* p_inputs)
{
    //shorter names for offset, convolution windows
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    uint8_t ofs_h = offset_window.h ;
    uint8_t ofs_w = offset_window.w ;
    uint8_t cw_h = conv.window.h ;
    uint8_t cw_w = conv.window.w ;
    uint64_t input_i, filt_i, output_i ;

    //make sure model was prepared
    assert(b_prep) ;

    //allocate output memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, OutDepth) ;

    uint32_t od;
    uint32_t ph;
    uint32_t pw;
    int idx;
    int sm_ctr;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for collapse(3) shared(st, p_output, p_inputs) private(sm_ctr, idx, flag, input_i, filt_i, output_i, ph, pw)
    for(od = 0 ; od < OutDepth ; od++)
    {    //output picture dimensions
        for(ph = 0 ; ph < out_hw.h ; ph++)
        {
            for(pw = 0 ; pw < out_hw.w ; pw++)
            {
                if (flag != 1) { // should execute once per CPU thread
                  idx = omp_get_thread_num();
                  cudaSetDevice(idx);
                  st[idx] = new Stream[sm_num];
                  for (int i = 0; i < sm_num; i++) {
                    st[idx][i].Create();
                  }
                  flag = 1;
                  sm_ctr = -1;
                }
                sm_ctr++;
                output_i = get_output_i(ph, pw, od) ;
                for(uint32_t di = 0 ; di < lay_dim.in_dep ; di++)
                {
                    //filters need extra checks to make sure we are in bounds
                    for(uint16_t fh = 0 ; (fh < cw_h) ; fh++)
                    {
                        if(conv.same_pad && ((uint16_t)(fh+ph-ofs_h) >= lay_dim.hw.h)){ continue ; }  //should underflow
                        for(uint16_t fw = 0 ; (fw < cw_w) ; fw++)
                        {
                            if(conv.same_pad && ((uint16_t)(fw+pw-ofs_w) >= lay_dim.hw.w)){ continue ; }  //should underflow
                            input_i = get_input_i((fh+ph-ofs_h), (fw+pw-ofs_w), di) ;
                            filt_i = get_filter_i(fh, fw, di, od) ;
                            //multiply_f accumulate
                            if(p_tern == NULL || p_tern[filt_i] == 0)
                            {
                                if (p_filters[filt_i] == 1) {
                                    IntOps::add(&p_output->enc_segs[idx][output_i], &p_output->enc_segs[idx][output_i], &p_inputs->enc_segs[idx][input_i], st[idx][sm_ctr % sm_num]);
                                }
                                else {
                                    IntOps::subtract(&p_output->enc_segs[idx][output_i], &p_output->enc_segs[idx][output_i], &p_inputs->enc_segs[idx][input_i], st[idx][sm_ctr % sm_num]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "IntConv: " << time_span.count() << " seconds." << std::endl;
    //free constants input data
    fixpt_free_global((lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep), p_inputs) ;
    return p_output ;
}

bool inline IntFunc::Convolution::retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
                                        uint32_t* di, uint32_t* fh, uint32_t* fw)
{
    bool oob = false ;
    uint32_t cw_area = conv.window.h * conv.window.w ;
    *di = i/cw_area ;
    *fh = (i%cw_area)/conv.window.w  ;
    *fw = i%conv.window.w  ;
    //bounds check 1: make sure it is in bounds of num_ops
    if(i >= (lay_dim.in_dep*cw_area)){ oob = true ; }
    //bounds check 2: for same padding, make sure filter is in bounds
    if(conv.same_pad)
    {
        if(((uint32_t)((*fh)+ph*conv.stride.h-offset_window.h) >= lay_dim.hw.h) ||
            ((uint32_t)((*fw)+pw*conv.stride.w-offset_window.w) >= lay_dim.hw.w))
        {  oob = true ; }
    }
    return oob ;
}

uint64_t inline IntFunc::Convolution::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}
uint64_t inline IntFunc::Convolution::get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od)
{
    return ((((fh)*conv.window.w + fw)*lay_dim.in_dep + di)*OutDepth + od) ;
}
uint64_t inline IntFunc::Convolution::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    return (((ph)*out_hw.w + pw)*OutDepth + od) ;
}

IntFunc::SumPooling::SumPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }
}

tDimensions* IntFunc::SumPooling::prep(tDimensions* ret_dim)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    assert((pool.stride.h) != 0 && pool.stride.w != 0);

    //set output depth
    if(pool.same_pad) //round up - (partial pools used)
    {
        out_hw.h = (lay_dim.hw.h - 1)/(pool.stride.h) + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/(pool.stride.w) + 1 ;
        if(pool.stride.h == 1){ offset_window.h = (int16_t)((pool.window.h-1)/2) ; }
        else{ offset_window.h = (out_hw.h*pool.stride.h - lay_dim.hw.h)/2 ; }
        if(pool.stride.w == 1){ offset_window.w = (int16_t)((pool.window.w-1)/2) ; }
        else{ offset_window.w = (out_hw.w*pool.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid pad - truncate (partial pools ignored)
    {
        offset_window.h = 0 ;
        offset_window.w = 0 ;
        out_hw.h = (lay_dim.hw.h-((uint16_t)(pool.window.h/2))-1)/pool.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w-((uint16_t)(pool.window.w/2))-1)/pool.stride.w + 1 ;
    }

    //update upper bound and out bits
    lay_dim.up_bound *= (pool.window.w) * (pool.window.h) ;
    for(lay_dim.out_bits = lay_dim.in_bits ;
        (lay_dim.up_bound >> lay_dim.out_bits) > 0 ;
        lay_dim.out_bits++) ;

    //update return dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale*(pool.window.w)*(pool.window.h) ;

    b_prep = true ;

    return ret_dim ;
}

tFixedPointPacked* IntFunc::SumPooling::execute(tFixedPointPacked* p_inputs)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    //input picture indexes
    tRectangle ip = {0,0} ; //input picture indexes
    uint64_t input_i, output_i ;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tFixedPointPacked* p_output;
    fixpt_calloc_global(&p_output, len, 1) ;
    uint32_t opw, oph, di;
    uint32_t fh, fw;
    int idx;
    int sm_ctr = 0;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for collapse(3) shared(st) private(flag, sm_ctr, idx, output_i, input_i, opw, oph, di, fh, fw)
    for(di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(opw = 0 ; opw < out_hw.w; opw++)
            {
                if (flag != 1) { // should execute once per CPU thread
                  idx = omp_get_thread_num();
                  cudaSetDevice(idx);
                  st[idx] = new Stream[sm_num];
                  for (int i = 0; i < sm_num; i++) {
                    st[idx][i].Create();
                  }
                  flag = 1;
                  Synchronize();
                  sm_ctr = -1;
                }
                    //get input picture indexes
                ip.h = oph * (pool.stride.h) - offset_window.h ;
                ip.w = opw * (pool.stride.w) - offset_window.w ;
                output_i = get_output_i(oph, opw, di) ;
                sm_ctr++;
                for(fh = 0 ; (fh < (pool.window.h))
                    && ((ip.h+fh) < (lay_dim.hw.h)) ; fh++)
                {
 		            if((ip.h + fh) < 0){ continue ; }
                    for(fw = 0 ; (fw < (pool.window.w))
                        && ((ip.w+fw) < lay_dim.hw.w) ; fw++)
                    {
 		                if((ip.w + fw) < 0){ continue ; }
                        input_i = get_input_i(ip.h+fh, ip.w+fw, di) ;
                        IntOps::add(&p_output->enc_segs[idx][output_i], &p_output->enc_segs[idx][output_i], &p_inputs->enc_segs[idx][input_i], st[idx][sm_ctr % sm_num]) ;
                        sm_ctr++;
                    }
                }
            }
        }
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "IntSumPool: " << time_span.count() << " seconds." << endl;
    return p_output ;
}

uint64_t inline IntFunc::SumPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}
uint64_t inline IntFunc::SumPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //indepth == outdepth
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}
//these functions should only be used by weight_convert program
void IntFunc::SumPooling::extract_bias(tFixedPointPacked* p_bias){ printf("Weight convert not defined\r\n") ; }

#define SLOPE_BITS 8
IntFunc::Quantize::Quantize(tQParams* qparam)
{
    dim_len = SIZE_EMPTY ;
    shift_bits = qparam->shift_bits ;
    b_prep = false ;
}


tDimensions* IntFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBitPacked* p_bias, uint16_t* p_slope)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;

    //get bias offset
    uint32_t bias_len = ret_dim->in_dep ;
    assert((fd_bias != NULL) && (p_bias != NULL)) ;
    BinOps::get_intfilters(fd_bias, (tMultiBitPacked**) &p_bias, bias_len) ;
    if(p_slope!=NULL && shift_bits> 1){ BinOps::get_intfilters_ptxt(fd_bias, p_slope, bias_len) ; }
    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    uint8_t sc_b = 0 ;
    for(sc_b= 0 ; (1<<sc_b) < lay_dim.scale ; sc_b++);
    slope_bits = SLOPE_BITS + sc_b - shift_bits ;

    //set output dimensions, other parameters
    if(shift_bits == 0) {
       lay_dim.out_bits = lay_dim.in_bits ;
       ret_dim->up_bound = lay_dim.up_bound ;
       ret_dim->scale = lay_dim.scale ;
    } else if(shift_bits==1) {
       lay_dim.out_bits = 1 ;
       ret_dim->up_bound = 1 ;
       ret_dim->scale = 0.5 ;
    } else {
       lay_dim.out_bits = shift_bits ;
       ret_dim->up_bound = (1<<(lay_dim.out_bits)) - 1 ;
       ret_dim->scale = ret_dim->up_bound ; 
    }
    dim_len = get_size(&lay_dim.hw, SIZE_EMPTY, lay_dim.in_dep) ;

    //update dimensions
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = (1<<(lay_dim.out_bits-1)) ;
    bit_calloc_global(&p_output, dim_len);

    b_prep = true ;
    fixpt_calloc_global(&x_add, 40, 1);
    fixpt_calloc_global(&x_bn, 1, FIXEDPOINT_BITS) ;
    fixpt_calloc_global(&p_output_relu, dim_len, FIXEDPOINT_BITS);
    return ret_dim ;
}

tBitPacked* IntFunc::Quantize::execute(tFixedPointPacked* p_inputs, tFixedPointPacked* p_bias)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    //input error checking
    assert(b_prep) ;

    uint32_t i;
    int idx;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    int sm_ctr = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for shared(st, x_add, p_output, p_inputs) firstprivate(sm_ctr) private(idx, flag)
    for(i = 0 ; i < dim_len ; i++)
    {
        if (flag != 1) { // should execute once per CPU thread
          idx = omp_get_thread_num();
          cudaSetDevice(idx);
          st[idx] = new Stream[sm_num];
          for (int i = 0; i < sm_num; i++) {
            st[idx][i].Create();
          }
          flag = 1;
          Synchronize();
          sm_ctr = 0;
        }
        uint32_t di = i % lay_dim.in_dep ;
        IntOps::add(&x_add->enc_segs[idx][sm_ctr%sm_num], &p_inputs->enc_segs[idx][i], &p_bias->enc_segs[idx][di], st[idx][sm_ctr % sm_num]);
        IntOps::binarize_int(&x_add->enc_segs[idx][sm_ctr%sm_num].ctxt[0], st[idx][sm_ctr % sm_num]);
        Copy(p_output->enc_segs[idx][i], x_add->enc_segs[idx][sm_ctr%sm_num].ctxt[0], st[idx][sm_ctr % sm_num]);
        sm_ctr++;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "IntQuant: " << time_span.count() << " seconds.\n";
    fixpt_free_global(1, x_add) ;
    fixpt_free_global(dim_len, p_inputs) ;
    return p_output ;
}

tFixedPointPacked* IntFunc::Quantize::add_bias(tFixedPointPacked* p_inputs, tMultiBitPacked* p_bias)
{
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  tMultiBitPacked* p_output ;

  //input error checking
  assert(b_prep) ;

  //allocate memory
  mbit_calloc_global(&p_output, dim_len,1) ;
  uint32_t i;
  int idx;
  uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
  uint8_t flag = 0;
  int st_ctr = 0;
  Stream* st[NUM_GPUS];
  omp_set_num_threads(NUM_GPUS);
  #pragma omp parallel for firstprivate(st_ctr) private(idx, flag) shared(st, p_inputs, p_bias, p_output)
  for(i = 0 ; i < dim_len ; i++)
  {
      if (flag != 1) { // should execute once per CPU thread
        idx = omp_get_thread_num();
        cudaSetDevice(idx);
        st[idx] = new Stream[sm_num];
        for (int i = 0; i < sm_num; i++) {
          st[idx][i].Create();
        }
        flag = 1;
      }
      uint32_t di = i % lay_dim.in_dep ;
      BinOps::int_add(p_output->enc_segs[idx][i].ctxt[0], p_inputs->enc_segs[idx][i].ctxt[0], p_bias->enc_segs[idx][di].ctxt[0], st[idx][st_ctr % sm_num]) ;
      st_ctr++;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "IntAddBias: " << time_span.count() << " seconds." << endl;
  mbit_free_global(dim_len, p_inputs) ;
  mbit_free_global(1, x_add) ;
  return p_output ;
}

tFixedPointPacked* IntFunc::Quantize::relu_shift(tFixedPointPacked* p_inputs, tMultiBitPacked* p_bias, uint16_t* p_slope)
{
    //input error checking
    assert(b_prep) ;
    int idx;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    int st_ctr = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);

    #pragma omp parallel for firstprivate(st_ctr) private(idx, flag) shared(st, p_inputs, p_bias, p_slope, p_output_relu, slope_bits)
    for(uint64_t i = 0 ; i < dim_len ; i++)
    {
        if (flag != 1) { // should execute once per CPU thread
          idx = omp_get_thread_num();
          cudaSetDevice(idx);
          st[idx] = new Stream[sm_num];
          for (int i = 0; i < sm_num; i++) {
            st[idx][i].Create();
          }
          flag = 1;
          Synchronize();
        }
        uint32_t di = i % lay_dim.in_dep ;
        IntOps::multiply_pc_ints(x_bn->enc_segs[idx][0].ctxt[0], p_inputs->enc_segs[idx][i].ctxt[0], &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, st[idx][st_ctr % sm_num]) ;
	    IntOps::add(&(x_bn->enc_segs[idx][0]), &(x_bn->enc_segs[idx][0]), &(p_bias->enc_segs[idx][i]), st[idx][st_ctr % sm_num]) ;
        IntOps::binarize_int(&(x_bn->enc_segs[idx][0].ctxt[0]), st[idx][st_ctr % sm_num]) ;
        IntOps::shift(&(p_output_relu->enc_segs[idx][i]), &(x_bn->enc_segs[idx][0]), slope_bits, shift_bits+1, st[idx][st_ctr % sm_num]) ;
        IntOps::relu(&(p_output_relu->enc_segs[idx][i]), &(p_output_relu->enc_segs[idx][i]), MULTIBIT_BITS, st[idx][st_ctr % sm_num]) ;
	    BinOps::unbinarize_int(p_output_relu->enc_segs[idx][i].ctxt[0], st[idx][st_ctr%sm_num]);
        st_ctr++;
    }

    fixpt_free_global(1, x_bn) ;
    fixpt_free_global(dim_len, p_inputs) ;

    return p_output_relu ;
}

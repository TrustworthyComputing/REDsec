#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <chrono>
#include <iostream>
#include "Layer.cuh"
#include <omp.h>
#include <cmath>
#include "BinFunc_gpu.cuh"
#include "BinOps_gpu.cuh"
#include "IntOps_gpu.cuh"

using namespace std;
using namespace std::chrono;
using namespace redcufhe;

BinFunc::Convolution::Convolution(uint32_t out_dep, tConvParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert(out_dep > 0) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    OutDepth = out_dep ;
    memcpy(&conv, in_params, sizeof(conv)) ;
    b_prep = false ;
}

tDimensions* BinFunc::Convolution::prep(FILE* fd_filt, tDimensions* ret_dim)
{
    assert(!b_prep) ;
    assert((ret_dim != NULL) && (fd_filt != NULL)) ;
    assert((conv.stride.h != 0) && (conv.stride.w != 0)) ;

    //set parameters depth
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    if(conv.same_pad)
    {   //zero padding -offset is half of window size
        out_hw.h = (lay_dim.hw.h-1)/conv.stride.h + 1 ;
        out_hw.w = (lay_dim.hw.w-1)/conv.stride.w + 1 ;
        if(conv.stride.h == 1){ offset_window.h = (int16_t)((conv.window.h-1)/2) ; }
	    else{ offset_window.h = (out_hw.h*conv.stride.h - lay_dim.hw.h)/2 ; }
        if(conv.stride.w == 1){ offset_window.w = (int16_t)((conv.window.w-1)/2) ; }
	    else{ offset_window.w = (out_hw.w*conv.stride.w - lay_dim.hw.w)/2 ; }
    }
    else //valid padding - no offest, but remove left and right borders
    {
        offset_window.w = 0 ;
        offset_window.h = 0 ;
        out_hw.h = lay_dim.hw.h - 2*((int16_t)((conv.window.h-1)/2)) ;
        out_hw.h = out_hw.h/conv.stride.h ;
        out_hw.w = lay_dim.hw.w - 2*((int16_t)((conv.window.w-1)/2)) ;
        out_hw.w = out_hw.w/conv.stride.w ;
    }

    //calculate number of additions for output bitsize
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
    p_filters = new uint8_t [flen] ;
    p_tern = new uint8_t [flen] ;

    //get filters
    BinOps::get_ternfilters(fd_filt, p_filters, p_tern, flen, conv.tern_thresh) ;

    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_dep = OutDepth ;
    ret_dim->in_bits = lay_dim.out_bits  ;
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale ;
    ret_dim->out_bits = SINGLE_BIT ;

    b_prep = true ;
    uint32_t partsum_ops = (conv.window.h)*(conv.window.w)*(lay_dim.in_dep) ;

    mbit_calloc_global(&p_window, partsum_ops, 1);
    mbit_calloc_global(&p_output, (lay_dim.hw.h)*(lay_dim.hw.w)*OutDepth, 1) ;
    bit_calloc_global(&p_inputs_bar, (lay_dim.hw.h)*(lay_dim.hw.w)*(lay_dim.in_dep));

    return ret_dim ;
}

tMultiBitPacked* BinFunc::Convolution::execute(tBitPacked* p_inputs)
{
    uint64_t input_i, filt_i, output_i ;
    uint8_t oob_i;

    //shorter names for offset, convolution windows
    int16_t ofs_h = offset_window.h ;
    int16_t ofs_w = offset_window.w ;

    //number of items per output
    uint32_t partsum_ops = (conv.window.h)*(conv.window.w)*(lay_dim.in_dep) ;

    //make sure model was prepared
    assert(b_prep) ;

    uint32_t od, ph, pw, fh, fw, di, wi;
    bool oob;
    int idx;
    uint64_t ops;
    int st_ctr = 0;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    #pragma omp parallel for collapse(3) shared(p_output) private(st, st_ctr, idx, flag, input_i, filt_i, output_i, ph, pw, fh, fw, di, oob, ops)
    for(ph = 0 ; ph < lay_dim.hw.h ; ph++)
    {
        for(pw = 0 ; pw < lay_dim.hw.w ; pw++)
        {
            for(di = 0 ; di < lay_dim.in_dep ; di++)
            {
                if (flag != 1) { // should execute once per CPU thread
                    idx = omp_get_thread_num();
                    cudaSetDevice(idx);
                    st_ctr = 0;
                    st[idx] = new Stream[sm_num];
                    for (int i = 0; i < sm_num; i++) {
                        st[idx][i].Create();
                    }
                    flag = 1;
                    Synchronize();
                }
                //Loop 1a: multiply
                assert(p_inputs_bar != NULL) ;
                input_i = get_input_i(ph, pw, di) ;
                //XNOR 0 is NOT
                BinOps::multiply(&p_inputs_bar->enc_segs[idx][input_i], &p_inputs->enc_segs[idx][input_i], 0, st[idx][st_ctr % sm_num]) ;
                //XNOR 1 is copy - use original input
                BinOps::unbinarize_int_inv(p_inputs_bar->enc_segs[idx][input_i], st[idx][st_ctr % sm_num]);
                BinOps::unbinarize_int(p_inputs->enc_segs[idx][input_i], st[idx][st_ctr % sm_num]);
                st_ctr++;
            }
        }
    }
    Synchronize();
    flag = 0;
    #pragma omp parallel for collapse(3) private(st_ctr, wi, idx, flag, input_i, oob_i, filt_i, output_i, fh, fw, di, oob, ops)
    for(od = 0 ; od < OutDepth ; od++)
    {   //output picture dimensions
        for(ph = 0 ; ph < out_hw.h ; ph++)
        {
            for(pw = 0 ; pw < out_hw.w ; pw++)
            {
                if (flag != 1) { // should execute once per CPU thread
                  idx = omp_get_thread_num();
                  cudaSetDevice(idx);
                  st_ctr = 0;
                  st[idx] = new Stream[sm_num];
                  for (int i = 0; i < sm_num; i++) {
                    st[idx][i].Create();
                  }
                  flag = 1;
                  Synchronize();
                }
		        oob_i = 0;
                for(wi = 0 ; wi < partsum_ops ; wi++)
                {
                    oob = retrieve_dims(wi, ph, pw, &di, &fh, &fw) ;
                    filt_i = get_filter_i(fh, fw, di, od) ;
                    input_i = get_input_i((fh+ph-ofs_h), (fw+pw-ofs_w), di) ;
                    if(!oob && ((p_tern==NULL) || (p_tern[filt_i] == 0)))
                    {
                        if(p_filters[filt_i] == 0)
                        {
                            for (int el = 0; el <= p_window->enc_segs[idx][wi].ctxt[0].lwe_sample_->n(); el ++)
                              p_window->enc_segs[idx][wi].ctxt[0].lwe_sample_->data()[el] = p_inputs_bar->enc_segs[idx][input_i].lwe_sample_->data()[el];
                        }
                        else
                        {
                            for (int el = 0; el <= p_window->enc_segs[idx][wi].ctxt[0].lwe_sample_->n(); el ++)
                              p_window->enc_segs[idx][wi].ctxt[0].lwe_sample_->data()[el] = p_inputs->enc_segs[idx][input_i].lwe_sample_->data()[el];
                        }
                    }
                    //padding, alternate 0s and 1s
                    else if(p_tern != NULL && p_tern[filt_i] == 1)
                    {
                       Torus mu = ModSwitchToTorus(-1, MSG_SPACE);
                       NoiselessTrivial(p_window->enc_segs[idx][wi].ctxt[0], mu);
                    }
                    else {
                        if(((oob_i++) % 2) == 0)
                        {
                            Torus mu = ModSwitchToTorus(-1, MSG_SPACE);
                            NoiselessTrivial(p_window->enc_segs[idx][wi].ctxt[0], mu);
                        }
                        else
                        {
                            Torus mu = ModSwitchToTorus(1, MSG_SPACE);
                            NoiselessTrivial(p_window->enc_segs[idx][wi].ctxt[0], mu);
                        }
                    }
                    //zero weight value, handled in bias
                }
                output_i = get_output_i(ph, pw, od) ;
                for (ops = 0; ops < partsum_ops; ops++) {
                  BinOps::int_add(p_output->enc_segs[idx][output_i].ctxt[0], p_output->enc_segs[idx][output_i].ctxt[0], p_window->enc_segs[idx][ops].ctxt[0], st[idx][st_ctr % sm_num]);
                  st_ctr = (st_ctr + 1) % 40;
                }
            }
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "BinConv: " << time_span.count() << " seconds." << std::endl;

    //free constants input data
    bit_free_global(p_inputs) ;
    bit_free_global(p_inputs_bar);
    mbit_free_global(partsum_ops, p_window);
    return p_output ;
}

bool inline BinFunc::Convolution::retrieve_dims(uint64_t i, uint32_t ph, uint32_t pw,
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

uint64_t inline BinFunc::Convolution::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}
uint64_t inline BinFunc::Convolution::get_filter_i(uint32_t fh, uint32_t fw, uint32_t di, uint32_t od)
{
    return ((((fh)*conv.window.w + fw)*lay_dim.in_dep + di)*OutDepth + od) ;
}
uint64_t inline BinFunc::Convolution::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    return (((ph)*out_hw.w + pw)*OutDepth + od) ;
}
//used for debugging
void BinFunc::Convolution::get_outhw(tRectangle* ret_hw)
{
    memcpy(ret_hw, &out_hw, sizeof(out_hw));
}
void BinFunc::Convolution::get_outdep(uint32_t* out_dep)
{
    *out_dep = OutDepth ;
}

//these functions should only be used by weight_convert program
void BinFunc::Convolution::extract_bias(FILE* fd_filt, tMultiBitPacked* p_bias, eBiasType e_bias){ printf("Weight convert not defined\r\n") ; }
void BinFunc::Convolution::export_weights(FILE* fd){ printf("Weight convert not defined\r\n") ; }

BinFunc::SumPooling::SumPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }
    b_prep = false ;
}


tDimensions* BinFunc::SumPooling::prep(tDimensions* ret_dim)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;
    assert((pool.stride.h) != 0 && pool.stride.w != 0) ;

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

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
    ret_dim->up_bound = lay_dim.up_bound  ;
    ret_dim->scale = lay_dim.scale * (pool.window.w) * (pool.window.h) ;
    ret_dim->out_bits = SINGLE_BIT ;

    b_prep = true ;
    return ret_dim ;
}

tMultiBitPacked* BinFunc::SumPooling::execute(tMultiBitPacked* p_inputs)
{
    //input picture indexes
    tRectangle ip = {0,0} ;
    uint64_t input_i, output_i ;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tMultiBitPacked* p_output;
    mbit_calloc_global(&p_output, len, MULTIBIT_BITS) ;
    uint32_t opw, oph, di;
    uint32_t fh, fw;
    int idx;
    int st_ctr = 0;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for collapse(3) shared(st) private(flag, idx, st_ctr, output_i, opw, oph, di, fh, fw)
    for(di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(opw = 0 ; opw < out_hw.w; opw++)
            {
                if (flag != 1) { // should execute once per CPU thread
                  idx = omp_get_thread_num();
                  cudaSetDevice(idx);
                  st_ctr = -1;
                  st[idx] = new Stream[sm_num];
                  for (int i = 0; i < sm_num; i++) {
                    st[idx][i].Create();
                  }
                  flag = 1;
                }
                //get input picture indexes
                ip.h = oph * (pool.stride.h) - offset_window.h ;
                ip.w = opw * (pool.stride.w) - offset_window.w ;
                output_i = get_output_i(oph, opw, di) ;
                st_ctr++;
                for(fh = 0 ; (fh < (pool.window.h))
                    && ((ip.h+fh) < (lay_dim.hw.h)) ; fh++)
                {
                    if((ip.h + fh) < 0){ continue ; }
                    for(fw = 0 ; (fw < (pool.window.w))
                        && ((ip.w+fw) < lay_dim.hw.w) ; fw++)
                    {
                     	if((ip.w + fw) < 0){ continue ; }
                        input_i = get_input_i(ip.h+fh, ip.w+fw, di) ;
                        BinOps::int_add(p_output->enc_segs[idx][output_i].ctxt[0], &p_output->enc_segs[idx][output_i].ctxt[0], &p_inputs->enc_segs[idx][input_i].ctxt[0], st[idx][st_ctr % sm_num]) ;
                    }
                }
            }
        }
    }

    mbit_free_global((lay_dim.hw.h * lay_dim.hw.w * lay_dim.in_dep), p_inputs) ;
    return p_output ;
}

uint64_t inline BinFunc::SumPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}
uint64_t inline BinFunc::SumPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //indepth == outdepth
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}

void BinFunc::SumPooling::get_outhw(tRectangle* ret_hw)
{
    memcpy(ret_hw, &out_hw, sizeof(out_hw));
}
void BinFunc::SumPooling::get_outdep(uint32_t* out_dep)
{
    *out_dep = lay_dim.in_dep ;
}

//these functions should only be used by weight_convert program
void BinFunc::SumPooling::extract_bias(tMultiBitPacked* p_bias){ printf("Weight convert not defined\r\n") ; }

BinFunc::MaxPooling::MaxPooling(tPoolParams* in_params)
{
    //error checking
    assert(in_params != NULL) ;
    assert((in_params->window.h > 0) && (in_params->window.w > 0));

    memcpy(&pool, in_params, sizeof(pool)) ;
    if(pool.stride.h == 0){ pool.stride.h = pool.window.h ; }
    if(pool.stride.w == 0){ pool.stride.w = pool.window.w ; }
    b_prep = false ;
}

tDimensions* BinFunc::MaxPooling::prep(tDimensions* ret_dim)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;
    assert(((pool.window.h) != 0) && ((pool.window.w) != 0)) ;
    assert((pool.stride.h) != 0 && pool.stride.w != 0) ;

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    //set output dimensions
    if(pool.same_pad) //round up - (partial pools used)
    {
        out_hw.h = (lay_dim.hw.h - 1)/(pool.stride.h) + 1 ;
        out_hw.w = (lay_dim.hw.w - 1)/(pool.stride.w) + 1 ;
    }
    else //valid pad - truncate (partial pools ignored)
    {
        out_hw.h = (lay_dim.hw.h)/(pool.window.h) ;
        out_hw.w = (lay_dim.hw.w)/(pool.window.w) ;
    }
    //max value stays the same
    lay_dim.out_bits = lay_dim.in_bits ;

    //update dimensions
    ret_dim->hw.h = out_hw.h ;
    ret_dim->hw.w = out_hw.w ;
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->scale = lay_dim.scale ;

    b_prep = true ;

    return ret_dim ;
}

tBitPacked* BinFunc::MaxPooling::execute(tBitPacked* p_inputs)
{
    //input picture indexes
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    tRectangle ip = {0,0} ;
    uint64_t input_i, output_i ;

    //input error checking
    assert(b_prep) ;

    //allocate memory
    uint32_t len = get_size(&out_hw, SIZE_EMPTY, lay_dim.in_dep) ;
    tBitPacked* p_output;
    bit_calloc_global(&p_output, len) ;

    uint32_t opw, oph, di;
    uint32_t fh, fw;
    int idx;
    int st_ctr = 0;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for collapse(3) shared(st, p_inputs, p_output) private(flag, st_ctr, idx, output_i, opw, oph, di, fh, fw)
    for(di = 0 ; di < lay_dim.in_dep ; di++)
    {
        for(oph = 0 ; oph < out_hw.h ; oph++)
        {
            for(opw = 0 ; opw < out_hw.w; opw++)
            {
                if (flag != 1) { // should execute once per CPU thread
                  idx = omp_get_thread_num();
                  cudaSetDevice(idx);
                  st_ctr = -1;
                  st[idx] = new Stream[sm_num];
                  for (int i = 0; i < sm_num; i++) {
                    st[idx][i].Create();
                  }
                  flag = 1;
                  Synchronize();
                }
                //get input picture indexes
                ip.h = oph * (pool.stride.h) - offset_window.h ;
                ip.w = opw * (pool.stride.w) - offset_window.w ;
		        output_i = get_output_i(oph, opw, di) ;
                st_ctr++;
                for(fh = 0 ; (fh < (pool.window.h))
                    && ((ip.h+fh) < (lay_dim.hw.h)) ; fh++)
                {
 		            if((ip.h + fh) < 0){ continue ; }
                    for(fw = 0 ; (fw < (pool.window.w))
                        && ((ip.w+fw) < lay_dim.hw.w) ; fw++)
                    {
 		                if((ip.w + fw) < 0){ continue ; }
                        input_i = get_input_i(ip.h+fh, ip.w+fw, di) ;
                        BinOps::max(&p_output->enc_segs[idx][output_i], &p_output->enc_segs[idx][output_i], &p_inputs->enc_segs[idx][input_i], st[idx][st_ctr % sm_num]) ;
                    }
                }
            }
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "BinMaxPool: " << time_span.count() << " seconds.\n";
    bit_free_global(p_inputs) ;
    return p_output ;
}

uint64_t inline BinFunc::MaxPooling::get_input_i(uint32_t ph, uint32_t pw, uint32_t di)
{
    return (((ph)*lay_dim.hw.w + pw)*lay_dim.in_dep + di) ;
}
uint64_t inline BinFunc::MaxPooling::get_output_i(uint32_t ph, uint32_t pw, uint32_t od)
{
    //indepth == outdepth
    return (((ph)*out_hw.w + pw)*lay_dim.in_dep + od) ;
}

 BinFunc::Quantize::Quantize(tQParams* qparam)
{
    assert(qparam!=NULL && qparam->shift_bits > 0);
    dim_len = SIZE_EMPTY ;
    shift_bits = qparam->shift_bits ;
    b_prep = false ;
}
#define SLOPE_BITS 16
tDimensions* BinFunc::Quantize::prep(FILE* fd_bias, tDimensions* ret_dim, tMultiBitPacked* p_bias, uint16_t* p_slope)
{
    //input error checking
    assert(!b_prep) ;
    assert((ret_dim != NULL)) ;

    //get bias offset
    uint32_t bias_len = ret_dim->in_dep ;
    assert((fd_bias != NULL) && (p_bias != NULL)) ;
    BinOps::get_intfilters(fd_bias, &p_bias, bias_len) ;
    if(p_slope != NULL)
    {
	    BinOps::get_intfilters_ptxt(fd_bias, p_slope, bias_len) ;
    }

    //copy dimension
    memcpy(&lay_dim, ret_dim, sizeof(lay_dim)) ;

    uint8_t sb = 0 ;
    for(sb= 0 ; (1<<sb) < sqrt(lay_dim.up_bound)/2 ; sb++);
    slope_bits = SLOPE_BITS + sb ;

    //set output dimensions, other parameters
    lay_dim.out_bits = (shift_bits>1)?(shift_bits+1):1 ;
    dim_len = get_size(&lay_dim.hw, SIZE_EMPTY, lay_dim.in_dep) ;

    //update dimensions
    ret_dim->in_bits = lay_dim.out_bits ;
    ret_dim->out_bits = SINGLE_BIT ;
    ret_dim->up_bound = (1<<(lay_dim.out_bits-1)) ;
    ret_dim->scale = (shift_bits>1) ? (ret_dim->up_bound) : 0.5 ;

    b_prep = true ;

    bit_calloc_global(&p_output, dim_len) ;
    mbit_calloc_global(&x_add, 1, 1) ;

    return ret_dim ;
}

tBitPacked* BinFunc::Quantize::execute(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //input error checking
    assert(b_prep) ;
    uint32_t i;
    int idx;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    int st_ctr = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for firstprivate(st_ctr) private(idx, flag) shared(st, x_add, p_inputs, p_bias, p_output)
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
        }
        uint32_t di = i % lay_dim.in_dep ;
        BinOps::int_add(x_add->enc_segs[idx][0].ctxt[0], p_inputs->enc_segs[idx][i].ctxt[0], p_bias->enc_segs[idx][di].ctxt[0], st[idx][st_ctr % sm_num]) ;
        BinOps::binarize_int(x_add->enc_segs[idx][0].ctxt[0], st[idx][st_ctr % sm_num]) ;
        Copy(p_output->enc_segs[idx][i], x_add->enc_segs[idx][0].ctxt[0], st[idx][st_ctr % sm_num]);
        st_ctr++;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "BinQuant: " << time_span.count() << " seconds." << endl;
    mbit_free_global(dim_len, p_inputs) ;
    mbit_free_global(1, x_add) ;
    return p_output ;
}

//these functions should only be used by weight_convert program
void BinFunc::Quantize::extract_bias(tMultiBitPacked* p_bias, uint16_t* p_slope){ printf("Weight convert not defined\r\n") ; }
void BinFunc::Quantize::export_weights(FILE* fd, tMultiBitPacked* p_bias, uint16_t* p_slope){ printf("Weight convert not defined\r\n") ; }

tMultiBitPacked* BinFunc::Quantize::add_bias(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias)
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
          Synchronize();
        }
        uint32_t di = i % lay_dim.in_dep ;
        BinOps::int_add(p_output->enc_segs[idx][i].ctxt[0], p_inputs->enc_segs[idx][i].ctxt[0], p_bias->enc_segs[idx][di].ctxt[0], st[idx][st_ctr % sm_num]) ;
        st_ctr++;
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "BinAddBias: " << time_span.count() << " seconds." << std::endl;
    mbit_free_global(dim_len, p_inputs) ;
    mbit_free_global(1, x_add) ;
    return p_output ;
}


tFixedPointPacked* BinFunc::Quantize::relu_shift(tMultiBitPacked* p_inputs, tMultiBitPacked* p_bias, uint16_t* p_slope)
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    assert(b_prep) ;
    int idx;
    uint32_t sm_num = 40; // CONFIG: set to number of streaming multiprocessors per GPU
    uint8_t flag = 0;
    int st_ctr = 0;
    Stream* st[NUM_GPUS];
    omp_set_num_threads(NUM_GPUS);
    #pragma omp parallel for firstprivate(st_ctr) private(idx, flag) shared(st, p_inputs, p_bias, p_slope, p_output)
    for(uint32_t i = 0 ; i < dim_len ; i++)
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
	    BinOps::multiply_pc_ints(x_bn->enc_segs[idx][0].ctxt[0], p_inputs->enc_segs[idx][i].ctxt[0], &(p_slope[di]), (lay_dim.in_bits), SLOPE_BITS, st[idx][st_ctr % sm_num]) ;
	    BinOps::int_add(x_fp->enc_segs[idx][0].ctxt[0], x_bn->enc_segs[idx][0].ctxt[0], p_bias->enc_segs[idx][di].ctxt[0], st[idx][st_ctr % sm_num]) ;
        BinOps::binarize_int(x_fp->enc_segs[idx][0].ctxt[0], st[idx][st_ctr % sm_num]) ;
        BinOps::shift(&(x_fp->enc_segs[idx][0]), &(x_fp->enc_segs[idx][0]), slope_bits, shift_bits+1, st[idx][st_ctr % sm_num]) ;
        BinOps::relu(&(p_output_relu->enc_segs[idx][i]), &(x_fp->enc_segs[idx][0]), shift_bits, st[idx][st_ctr % sm_num]) ;
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "BinReLU: " << time_span.count() << " seconds." << std::endl;
    mbit_free_global(dim_len, p_inputs) ;
    mbit_free_global(1, x_bn) ;

    return p_output_relu ;
}

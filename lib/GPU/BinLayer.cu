#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstring>
#include "Layer.cuh"
#include "BinFunc_gpu.cuh"
#include "BinLayer.cuh"

void BinLayer::set_version(uint8_t v, tNetParams* net)
{
    if(v<1)
    {
        net->conv.stride.h = 1 ;
        net->conv.stride.w = 1 ;
        net->pool.stride.h = 0 ;
        net->pool.stride.w = 0 ;
    }
}

BinLayer::BinLayer(eConvType ec, uint16_t dep,
                ePoolType ep,
                eQuantType eq,
                tNetParams* np)
{
    assert((ec < NUM_CONVS) && (np->e_bias < NUM_BIASES) && (ep < NUM_POOLS)) ;
    assert((np != NULL)) ;

    tNetParams net_param ;
    memcpy(&net_param, np, sizeof(tNetParams)) ;
    e_conv = ec ;
    conv_depth = dep ;
    e_bias = np->e_bias ;
    e_pool = ep ;
    e_activation = eq ;
    p_bias = NULL ; //to be allocated
    p_slope = NULL ;
    set_version(net_param.version, &net_param) ;

	//Layers - Convolution
    if((e_conv == E_FC_FINAL) || (e_conv == E_FC))
    {
        net_param.conv.window.h = 1 ;
    	net_param.conv.window.w = 1 ;
    	net_param.conv.same_pad = true ;
    }
    lconv = new BinFunc::Convolution(conv_depth, &(net_param.conv)) ;
   	//pooling
    if(e_pool == E_SUMPOOL){ lpool.s = new BinFunc::SumPooling(&(net_param.pool)) ; }
    else if(e_pool == E_MAXPOOL)
    {
	    assert(eq == E_ACTIVATION_SIGN) ;
	    lpool.m = new BinFunc::MaxPooling(&(net_param.pool)) ;
    }
   	//Quantizing
    if((e_activation == E_ACTIVATION_SIGN) || (net_param.quant.shift_bits < 1)){ net_param.quant.shift_bits = 1 ; }
    lquant = new BinFunc::Quantize(&(net_param.quant)) ;

#ifdef _WEIGHT_CONVERT_
    //Batch Norm
    if(e_bias == E_BNORM){ bnorm = new BinFunc::BatchNorm(&(net_param.bnorm)) ; }
    else{ bnorm = NULL ; }
#endif
}

tDimensions* BinLayer::prep(FILE* fd, tDimensions* dim)
{
    //allocate bias array
    mbit_calloc_global(&p_bias, conv_depth, 1) ;
    p_slope = NULL ;
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM)){ p_slope = (uint16_t*)calloc(conv_depth,sizeof(uint16_t)) ; }

	//initialize arguments
    tActParams pact ;
    pact.d = dim ;
    //save/calculate dimensions and extract weights
    memcpy(&in_dim, pact.d, sizeof(in_dim)) ;
    pact.d = (tDimensions*) BinLayer::run(E_PREP, &pact, fd) ;
    memcpy(&out_dim, pact.d, sizeof(out_dim)) ;

#ifdef _WEIGHT_CONVERT_
    tActParams dummy ;
    void* d = BinLayer::run(E_PREP_BIAS, &dummy, fd) ;
#endif
    return pact.d ;
}

tBitPacked* BinLayer::execute(tBitPacked* pb_in)
{
    tActParams pact ;
    pact.b = pb_in ;
    return (tBitPacked*) BinLayer::run(E_EXEC, &pact, NULL) ;
}

#ifdef _WEIGHT_CONVERT_
void BinLayer::export_weights(FILE* fd_bin)
{
    tActParams dummy ;
    BinLayer::run(E_EXPORT, &dummy, fd_bin) ;
}
#else
void BinLayer::export_weights(FILE* fd_bin){ printf("Weight convert no defined\r\n"); }
#endif

void* BinLayer::run(eAction e_act, tActParams* ap, FILE* fd)
{
    tBitPacked* bdata  = (tBitPacked*) ap->b ;
    tDimensions* dim = ap->d ;
    tMultiBitPacked* mdata ;
    //First is convolution
    switch(e_conv)
    {
        case E_FC_FINAL:
        case E_FC:
        {
            if(e_act == E_PREP)
            {
                //flatten
                dim->in_dep *= dim->hw.h * dim->hw.w ;
                dim->hw.h = 1 ;
                dim->hw.w = 1 ;
             }
        }
        //fall through
        case E_CONV:
        {
            if(e_act == E_PREP){ dim = lconv->prep(fd, dim) ; }
            else if(e_act == E_EXEC){ mdata = lconv->execute(bdata) ; }
            //weight prep/export
            else if(e_act == E_PREP_BIAS){ lconv->extract_bias(fd, p_bias, e_bias) ; }
            else if(e_act == E_EXPORT){ lconv->export_weights(fd) ; }
        }
        break ;
        default: { assert(0) ;}
        break ;
    }
    //Sumpooling
    if(e_pool == E_SUMPOOL)
    {
        if(e_act == E_PREP){ dim = lpool.s->prep(dim) ; }
        else if(e_act == E_EXEC){ mdata = lpool.s->execute(mdata) ; }
        //weight prep
        else if(e_act == E_PREP_BIAS){ lpool.s->extract_bias(p_bias) ; }
    }
#ifdef _WEIGHT_CONVERT_
    if(e_bias == E_BNORM)
    {
        if(e_act == E_PREP){ dim = bnorm->prep(dim) ; }
        if(e_act == E_PREP_BIAS){ bnorm->extract_bias(fd, p_bias, p_slope) ; }
    }
#endif

    //quantize
    if(e_act == E_PREP)
    {
      dim = lquant->prep(fd, dim, p_bias, p_slope) ;
    }
    else if(e_act == E_EXEC)
    {
      if(e_activation == E_ACTIVATION_SIGN)
      {
          bdata = lquant->execute(mdata, p_bias) ;
      }else if(e_activation == E_ACTIVATION_NONE)
      {
          mdata = lquant->add_bias(mdata, p_bias) ;
      }else if(e_activation == E_ACTIVATION_RELU)
      {
          mdata = (tMultiBitPacked*) lquant->relu_shift(mdata, p_bias, p_slope) ;
      }
    }

    //Max Pooling
    if(e_activation == E_ACTIVATION_SIGN)
    {
      if(e_pool == E_MAXPOOL)
      {
          if(e_act == E_PREP){ dim = lpool.m->prep(dim) ; }
          else if(e_act == E_EXEC){ bdata = lpool.m->execute(bdata) ; }
      }
    }

#ifdef _PRINT_LAYER_
    if(b_print_layer)
    {
        if(e_act == E_EXEC){ print_layer_out(bdata, &out_dim) ; }
        b_print_layer = false ;
    }
#endif
    //return
    if(e_act == E_PREP){ return (void*) dim ; }
    else if(e_act == E_EXEC)
    {
        if(e_activation != E_ACTIVATION_SIGN){ return (void*) mdata ; }
        else{ return (void*) bdata ; }
    }

    return (void*) NULL ;
}

#ifdef _PRINT_LAYER_
void BinLayer::set_print_layer(uint8_t i)
{
    char fname[] = "outX.txt" ;
    fname[3] = i + '0' ;
    fdebug = fopen(fname, "w");
    b_print_layer = true ;
}

void BinLayer::print_layer_mid(tMultiBit* p_inputs, tDimensions* dim)
{
    if(print_label[0] == '\0')
    {
        strncpy(print_label, "Lay", sizeof(print_label)) ;
    }

    tMultiBit (*inputs)[dim->hw.h][dim->hw.w][dim->in_dep] =
            (tMultiBit (*) [dim->hw.h][dim->hw.w][dim->in_dep]) p_inputs ;
    for(int ph = 0 ; ph < dim->hw.h ; ph++)
    {
        for(int pw = 0 ; pw < dim->hw.w ; pw++)
        {
            fprintf(fdebug, "%s[%2d][%2d]:",print_label, ph, pw) ;
            for(uint32_t di = 0 ; di < dim->in_dep ; di++)
            {
                fprintf(fdebug, "%4d ", (int) (*inputs)[ph][pw][di]) ;
            }
            fprintf(fdebug, "\r\n") ;
        }
    }
}
void BinLayer::print_layer_out(tBit* p_inputs, tDimensions* dim)
{
    if(print_label[0] == '\0')
    {
        strncpy(print_label, "Lay", sizeof(print_label)) ;
    }
    tBit (*inputs)[dim->hw.h][dim->hw.w][dim->in_dep] =
            (tBit (*) [dim->hw.h][dim->hw.w][dim->in_dep]) p_inputs ;
    for(int ph = 0 ; ph < dim->hw.h ; ph++)
    {
        for(int pw = 0 ; pw < dim->hw.w ; pw++)
        {
            fprintf(fdebug, "%s[%2d][%2d]:",print_label, ph, pw) ;
            for(uint32_t di = 0 ; di < dim->in_dep ; di++)
            {
                fprintf(fdebug, "%d", (int) ((*inputs)[ph][pw][di])) ;
            }
            fprintf(fdebug, "\r\n") ;
        }
    }
}
#else
void BinLayer::set_print_layer(uint8_t i)
{
    b_print_layer = false ;
}
#endif

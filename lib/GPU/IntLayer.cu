#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstring>
#include "Layer.cuh"
#include "IntFunc_gpu.cuh"
#include "BinFunc_gpu.cuh"
#include "IntLayer.cuh"

void IntLayer::set_version(uint8_t v, tNetParams* net)
{
    if(v<1)
    {
        net->conv.stride.h = 1 ;
        net->conv.stride.w = 1 ;
        net->pool.stride.h = 0 ;
        net->pool.stride.w = 0 ;
    }
}

IntLayer::IntLayer(eConvType ec, uint16_t dep,
    ePoolType ep, eQuantType eq, tNetParams* np)
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

    set_version(net_param.version, &net_param) ;

    if((e_conv == E_FC_FINAL) || (e_conv == E_FC))
    {
        net_param.conv.window.h = 1 ;
        net_param.conv.window.w = 1 ;
        net_param.conv.same_pad = true ;
    }

    //Layers - Convolution
    lconv = new IntFunc::Convolution(conv_depth, &(net_param.conv)) ;
   	//Pooling
    if(e_pool == E_SUMPOOL){ lpool.s = new IntFunc::SumPooling(&(net_param.pool)) ; }
    else if(e_pool == E_MAXPOOL)
    {
	    assert(eq == E_ACTIVATION_SIGN) ;
	    lpool.m = new BinFunc::MaxPooling(&(net_param.pool)) ;
    }
    //Quantizing
    if((e_activation == E_ACTIVATION_SIGN)){ net_param.quant.shift_bits = 1 ; }
    else if(e_activation == E_ACTIVATION_NONE){ net_param.quant.shift_bits = 0 ; }
    lquant = new IntFunc::Quantize(&(net_param.quant)) ;
}

tDimensions* IntLayer::prep(FILE* fd, tDimensions* dim)
{
    //allocate bias array
    fixpt_calloc_global(&p_bias, conv_depth, 1) ;
    if((e_activation == E_ACTIVATION_RELU) && (e_bias == E_BNORM))
    {    p_slope = (uint16_t*)calloc(conv_depth,sizeof(uint16_t)); }

	//initialize arguments
    tActParams pact ;
    pact.d = dim ;
    //save/calculate dimensions and extract weights
    memcpy(&in_dim, pact.d, sizeof(in_dim)) ;
    pact.d = (tDimensions*) IntLayer::run(E_PREP, &pact, fd) ;
    memcpy(&out_dim, pact.d, sizeof(out_dim)) ;

    return pact.d ;
}

tBitPacked* IntLayer::execute(tMultiBitPacked* pfp_in)
{
    tActParams pact ;
    pact.fp = pfp_in ;
    return (tBitPacked*) IntLayer::run(E_EXEC, &pact, NULL) ;
}

void IntLayer::export_weights(FILE* fd_bin)
{
    tActParams dummy ;
    void* d = IntLayer::run(E_EXPORT, &dummy, fd_bin) ;
}

void* IntLayer::run(eAction e_act, tActParams* ap, FILE* fd)
{
    assert(ap != NULL) ;
    tFixedPointPacked* fpdata  = ap->fp ;
    tBitPacked* bdata ;

    tDimensions* dim = ap->d ;

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
            else if(e_act == E_EXEC){ fpdata = lconv->execute(fpdata) ; }
        }
        break ;
	      case E_NO_CONV:
	      break ;
        default: { assert(0) ;}
        break ;
    }

    //Sumpooling
    if(e_pool == E_SUMPOOL)
    {
        if(e_act == E_PREP){ dim = lpool.s->prep(dim) ; }
        else if(e_act == E_EXEC){ fpdata = lpool.s->execute(fpdata) ; }
        else if(e_act == E_PREP_BIAS){ lpool.s->extract_bias(p_bias) ; }
    }

    //quantize
    if(e_act == E_PREP){
      dim = lquant->prep(fd, dim, p_bias, p_slope) ;
    }
    else if(e_act == E_EXEC)
    {
      if(e_activation == E_ACTIVATION_SIGN){bdata = lquant->execute(fpdata, p_bias) ; }
      else if(e_activation == E_ACTIVATION_NONE){ fpdata = lquant->add_bias(fpdata, p_bias) ; }
      else if(e_activation == E_ACTIVATION_RELU){ fpdata = lquant->relu_shift(fpdata, p_bias, p_slope) ; }
    }
    //Max Pooling
    if(e_conv != E_FC_FINAL)
    {
      if(e_pool == E_MAXPOOL)
      {
	      if(e_activation != E_ACTIVATION_SIGN){ printf("Unexpected MaxPooling Operator on Integers\r\n") ; }
	      else if(e_act == E_PREP){ dim = lpool.m->prep(dim) ; }
          else if(e_act == E_EXEC){ bdata = lpool.m->execute(bdata) ; }
      }
    }

#ifdef _PRINT_LAYER_
    if(b_print_layer)
    {
        if(e_act == E_EXEC)
        {
            print_layer_out(fpdata, &out_dim) ;
        }
        b_print_layer = false ;
    }
#endif

    if(e_act == E_PREP){ return (void*) dim ; }
    else if(e_activation == E_ACTIVATION_SIGN && e_act == E_EXEC){ return (void*) bdata ; }
    else if(e_act == E_EXEC){ return (void*) fpdata ; }
    return (void*) NULL ;
}

#ifdef _PRINT_LAYER_
void IntLayer::set_print_layer(uint8_t i)
{
    char fname[] = "outX.txt" ;
    fname[3] = i + '0' ;
    fdebug = fopen(fname, "w");
    b_print_layer = true ;
}

void IntLayer::print_layer_out(tFixedPoint* p_inputs, tDimensions* dim)
{

    if(print_label[0] == '\0')
    {
        strncpy(print_label, "Lay0", sizeof(print_label)) ;
    }

    tFixedPoint (*inputs)[dim->hw.h][dim->hw.w][dim->in_dep] =
            (tFixedPoint (*) [dim->hw.h][dim->hw.w][dim->in_dep]) p_inputs ;

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
void IntLayer::set_print_layer(uint8_t i)
{
    b_print_layer = false ;
}
#endif

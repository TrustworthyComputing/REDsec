#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>

#if defined(GPU_ENC)
#include "net.cuh"
#include "GPU/Layer.cuh"
#include "GPU/BinLayer.cuh"
#include "GPU/IntLayer.cuh"
#else
#include "net.h"
#include "Layer.h"
#include "BinLayer.h"
#include "IntLayer.h"
#endif
//TODO: Add in your own network file. 
//MNIST is shown here as an example
#include "../mnist.h"
//End TODO: Add in your own network file.

HeBNN::HeBNN(FILE* in_file, bool b_prep)
{
	HeBNN::init(in_file, false) ;
}

HeBNN::HeBNN()
{
	//open up input file
	FILE *fd = fopen("var_prep.dat", "r");
	if(fd == NULL)
	{
		printf("Bad Weights File. Exiting...\r\n") ;
		return ;
	}
	HeBNN::init(fd, false) ;
	fclose(fd) ;
}

void HeBNN::init(FILE* in_file, bool b_prep)
{
	tDimensions* p_dim ;
	tDimensions lay_dim ;
	uint32_t conv_dep ;
	// read eval key
	#if defined(ENCRYPTED)
	//TODO: Modify location of evaluation key if needed
	FILE* cloud_key = fopen("../../../client/eval.key","rb");
	//End TODO: Modify location of evaluation key if needed
	bk = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
	fclose(cloud_key);
	#elif defined(GPU_ENC)
	//TODO: Modify location of evaluation key if needed
	ReadPubKeyFromFile(bk, "../../../client/eval.key");
	// End TODO: 
	omp_set_num_threads(NUM_GPUS);
	const auto processor_count = std::thread::hardware_concurrency();
	if (NUM_GPUS > processor_count) {
		printf("REDsec does not support configurations where the number of utilized GPUs "
					 "exceeds the number of possible CPU threads. Please reduce NUM_GPUs "
					 "in lib/GPU/Layer.cuh to reflect a value equal to or less than the "
					 "number of CPU threads in your system.");
		exit(0);
	}
	#pragma omp parallel for
	for (int i = 0; i < NUM_GPUS; i++) {
		cudaSetDevice(i);
		Initialize(bk);
	}
	#else
	bk = NULL;
	#endif
	
	//model netlist
	tNetParams params ;
	//dummy params
	params.pool.window.h = 2 ;
        params.pool.window.w = 2 ;
        params.pool.stride.h = 2 ;
        params.pool.stride.w = 2 ;
        params.pool.same_pad = false ;
        params.e_bias = E_NO_BIAS ;
        conv_dep =  SIZE_EMPTY ;
        params.conv.window.h = 1 ;
        params.conv.window.w = 1 ;
        params.conv.stride.h = 1 ;
        params.conv.stride.w = 1 ;
        params.conv.tern_thresh = 0.05 ;
        params.bnorm.use_scale = false ;
        params.bnorm.eps = 0.001 ;
        params.e_bias = E_BNORM ;

	//START: Auto generated model
	  //Dimensions of input
	lay_dim.hw.h = 28 ;
	lay_dim.hw.w = 28 ;
	lay_dim.in_dep = 1 ;
	lay_dim.in_bits = 4 ;
	lay_dim.out_bits = SINGLE_BIT ;
	lay_dim.filter_bits = SINGLE_BIT ; ;
	lay_dim.bias_bits = SINGLE_BIT ;
	//Start:TODO: Update to reflect preprocessing
	  //up_bound is the range of pixel values 
	lay_dim.up_bound = 2*15 ;
	  //scale is the mulitplicitive factor to map from
	  //  floating point to integer
	lay_dim.scale = 15 ;
	//End:TODO: Update to reflect preprocessing ;


	  //Model Params and Architechture

		//Layer 0
	params.pool.window.h = 2 ;
	params.pool.window.w = 2 ;
	params.pool.stride.h = 2 ;
	params.pool.stride.w = 2 ;
	params.pool.same_pad = false ;
	params.e_bias = E_NO_BIAS ;
	conv_dep =  SIZE_EMPTY ;
#if defined(GPU_ENC)
	layer0 = new IntLayer(E_NO_CONV, conv_dep, E_SUMPOOL, E_ACTIVATION_NONE, &params);
#else
	layer0 = new IntLayer(E_NO_CONV, conv_dep, E_SUMPOOL, E_ACTIVATION_NONE, &params, bk) ;
#endif

		//Layer 1
	params.conv.window.h = 1 ;
	params.conv.window.w = 1 ;
	params.conv.stride.h = 1 ;
	params.conv.stride.w = 1 ;
	params.conv.tern_thresh = 0.1 ;
	params.bnorm.use_scale = false ;
	params.bnorm.eps = 0.001 ;
	params.e_bias = E_BNORM ;
	params.quant.shift_bits = 4 ;
	conv_dep =  1024 ;
#if defined(GPU_ENC)
	layer1 = new IntLayer(E_FC, conv_dep, E_NO_POOL, E_ACTIVATION_RELU, &params);
#else
	layer1 = new IntLayer(E_FC, conv_dep, E_NO_POOL, E_ACTIVATION_RELU, &params, bk) ;
#endif

		//Layer 2
	params.e_bias = E_NO_BIAS ;
	conv_dep =  10 ;
#if defined(GPU_ENC)
	layer2 = new IntLayer(E_FC, conv_dep, E_NO_POOL, E_ACTIVATION_NONE, &params);
#else
	layer2 = new IntLayer(E_FC, conv_dep, E_NO_POOL, E_ACTIVATION_NONE, &params, bk) ;
#endif
	//END: Auto generated model
	
	//extract weights and initialize the dimensions of the model
	//START: Model prep
	p_dim = &lay_dim ;
	p_dim = (tDimensions*) layer0->prep(in_file, p_dim) ;
	p_dim = (tDimensions*) layer1->prep(in_file, p_dim) ;
	p_dim = (tDimensions*) layer2->prep(in_file, p_dim) ;
	//END: Model prep

}


#if defined(ENCRYPTED)
tMultiBit* HeBNN::run(tMultiBit* in_data)
#elif defined(GPU_ENC)
tMultiBitPacked* HeBNN::run(tMultiBitPacked* in_data)
#else
tMultiBit* HeBNN::run(tFixedPoint* in_data)
#endif
{
	#if defined(GPU_ENC)
	tMultiBitPacked* mbdata ;
	tBitPacked* bdata ;
	#else
	tBit* bdata ;
	tMultiBit* mbdata ;
	#endif

	//START: Execute inference model
#if !defined(GPU_ENC)
	mbdata = (tMultiBit*)layer0->execute(in_data) ;
#else
	mbdata = (tMultiBitPacked*)layer0->execute(in_data) ;
#endif
#if !defined(GPU_ENC)
	mbdata = (tMultiBit*)layer1->execute((tFixedPoint*) mbdata) ;
#else
	mbdata = (tMultiBitPacked*)layer1->execute((tMultiBitPacked*) mbdata) ;
#endif
#if !defined(GPU_ENC)
	mbdata = (tMultiBit*)layer2->execute((tFixedPoint*) mbdata) ;
#else
	mbdata = (tMultiBitPacked*)layer2->execute((tMultiBitPacked*) mbdata) ;
#endif
	//END: Execute inference model
	
	#if defined(GPU_ENC)
	return (tMultiBitPacked*)mbdata ;
	#else
	return (tMultiBit*)mbdata ;
	#endif
}

#ifdef _WEIGHT_CONVERT_
void HeBNN::export_weights(FILE* out_file)
{
	//START: Weight convert
layer0->export_weights(out_file) ;
layer1->export_weights(out_file) ;
layer2->export_weights(out_file) ;
	//END: Weight convert
	
	print_status("Exported Success\r\n") ;

	return ;
}
#else
void export_weights(FILE* out_file){ printf("Weight Convert not defined\r\n") ; }
#endif


void HeBNN::get_in_dims(tDimensions* in)
{
	if(in != NULL)
	{
		memcpy(in, &(layer0->in_dim), sizeof(*in)) ;
	}
}

void HeBNN::get_out_dims(tDimensions* out)
{
	if(out != NULL)
	{
		//START: Get out dimensions
		memcpy(out, &(layer2->out_dim), sizeof(*out)) ;
		//END: Get out dimensions
	}
}


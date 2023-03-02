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
	//END: Auto generated model
	
	//extract weights and initialize the dimensions of the model
	//START: Model prep
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
		//END: Get out dimensions
	}
}


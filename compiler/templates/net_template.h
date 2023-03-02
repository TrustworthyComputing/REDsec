#ifndef _NET_H_
#define _NET_H_

#include <vector>

#if defined(GPU_ENC)
#include "../../../lib/GPU/Layer.cuh"
#include "../../../lib/GPU/BinLayer.cuh"
#include "../../../lib/GPU/IntLayer.cuh"
#else
#include "../../../lib/Layer.h"
#include "../../../lib/BinLayer.h"
#include "../../../lib/IntLayer.h"
#endif
class HeBNN
{
	public:
		HeBNN() ;
		HeBNN(FILE* in_file, bool b_prep) ;
		void init(FILE* in_file, bool b_prep) ;
		#if defined(GPU_ENC)
		tMultiBitPacked* run(tMultiBitPacked* indata) ;
		#elif defined(ENCRYPTED)
		tMultiBit* run(tMultiBit* indata) ;
		#else
		tMultiBit* run(tFixedPoint* indata) ;
		#endif
		void get_in_dims(tDimensions* in) ;
		void get_out_dims(tDimensions* out) ;
		void export_weights(FILE* out_file) ;

		#if defined(GPU_ENC)
		redcufhe::PubKey bk;
		#else
		TFheGateBootstrappingCloudKeySet* bk;
		#endif
	private:
		//START layer generation
		//END layer generation
} ;
#endif

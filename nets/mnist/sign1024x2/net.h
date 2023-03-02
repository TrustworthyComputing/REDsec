#ifndef _NET_H_
#define _NET_H_

#include <vector>
#include "../../../lib/Layer.h"
#include "../../../lib/BinLayer.h"
#include "../../../lib/IntLayer.h"
class HeBNN
{
	public:
		HeBNN() ;
		HeBNN(FILE* in_file, bool b_prep) ;
		void init(FILE* in_file, bool b_prep) ;
		#if defined(ENCRYPTED)
		tMultiBit* run(tMultiBit* indata) ;
		#else
		tMultiBit* run(tFixedPoint* indata) ;
		#endif
		void get_in_dims(tDimensions* in) ;
		void get_out_dims(tDimensions* out) ;
		void export_weights(FILE* out_file) ;
		TFheGateBootstrappingCloudKeySet* bk;
	private:
		IntLayer* first_layer ;
		std::vector<BinLayer*> layers ;
		BinLayer* last_layer ;
} ;
#endif

#ifndef _NET_H_
#define _NET_H_

#include <vector>
#include "../../../lib/GPU/Layer.cuh"
#include "../../../lib/GPU/BinLayer.cuh"
#include "../../../lib/GPU/IntLayer.cuh"
class HeBNN
{
	public:
		HeBNN() ;
		HeBNN(FILE* in_file, bool b_prep) ;
		void init(FILE* in_file, bool b_prep) ;
		tMultiBitPacked* run(tMultiBitPacked* indata) ;
		void get_in_dims(tDimensions* in) ;
		void get_out_dims(tDimensions* out) ;
		void export_weights(FILE* out_file) ;
		redcufhe::PubKey bk;
	private:
		IntLayer* first_layer ;
		std::vector<BinLayer*> layers ;
		BinLayer* last_layer ;
} ;
#endif

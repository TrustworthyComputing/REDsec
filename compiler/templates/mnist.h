#ifndef _MNIST_H_
#define _MINST_H_
/*********************************** Defines ************************************/

#define PIC_SIZE 	 28
#define IN_CHANNELS   1
#define PIXEL_BITS	  8 //pixels are 8-bit numbers

#define CAT_0 NUM_0
/****************************** Structs And Enums *******************************/
typedef enum _MNIST
{
	RESULT_ERROR = -1,
	NUM_0 = 0,
	NUM_1,
	NUM_2,
	NUM_3,
	NUM_4,
	NUM_5,
	NUM_6,
	NUM_7,
	NUM_8,
	NUM_9,
}eMnist ;

/**************************** Function Declarations *****************************/
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "net.h"

/*********************************** Defines ************************************/
#define PREP_FILE true
#define RAW_FILE  false
/****************************** Structs And Enums *******************************/
/**************************** Function Declarations *****************************/
/******************************** Start of Code *********************************/
/*********************************************************************************
*    Func:
*    Desc:
*    Inputs:
*    Return:    None
*    Notes:
*********************************************************************************/
int main(void)
{
            //csv input
    FILE *fd_import = NULL ;
    FILE *fd_export = NULL ;
    
            //Open File
    fd_import = fopen("var.dat1", "r");
    fd_export = fopen("var_prep.dat", "w");
    if((fd_import == NULL) || (fd_export == NULL))
    {
        printf("Bad Input File. Exiting...\r\n") ;
        return -1 ;
    }   
        //create network
    HeBNN* network = new HeBNN(fd_import, RAW_FILE) ;
    network->export_weights(fd_export) ;
    
    fclose(fd_import) ;
    fclose(fd_export) ;
}


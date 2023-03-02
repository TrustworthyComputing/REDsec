#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "REDcuFHE/redcufhe_gpu.cuh"

using namespace std;
using namespace redcufhe;

int main(int argc, char** argv) {

    printf("Generating secure random keyset with 110 bits of security...\n");

    // generate keypair
    PriKey secret_key;
    PubKey public_key;
    PriKeyGen(secret_key);
    PubKeyGen(public_key, secret_key);

    // export secret key to file
    WritePriKeyToFile(secret_key, "secret.key");

    // export cloud key to file
    WritePubKeyToFile(public_key, "eval.key");

    printf("Finished exporting!\n");

    return 0;
}

#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#include <tfhe/tfhe_garbage_collector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

TFheGateBootstrappingParameterSet* redsec_params(){
    int lwe_dim = 630;
    int tlwe_dim = 1024;
    double std_dev = pow(2.,-40);
    double bootkey_std_dev = pow(2.,-41);
    double kskey_std_dev = pow(2., -35);
    // double std_dev = pow(2.,-160);
    // double bootkey_std_dev = pow(2.,-160); 
    // double kskey_std_dev = pow(2.,-160);


    LweParams* lwe_params = new_LweParams (lwe_dim, kskey_std_dev, std_dev);
    TLweParams* tlwe_params = new_TLweParams(tlwe_dim, 1, bootkey_std_dev, std_dev);
    TGswParams* tgsw_params = new_TGswParams(3, 10, tlwe_params);

    TfheGarbageCollector::register_param(lwe_params);
    TfheGarbageCollector::register_param(tlwe_params);
    TfheGarbageCollector::register_param(tgsw_params);

    return new TFheGateBootstrappingParameterSet(18, 1, lwe_params, tgsw_params);
}

int main(int argc, char** argv) {
    TFheGateBootstrappingParameterSet* params = redsec_params();

    // read 12 bytes from /dev/urandom
    FILE* urandom = fopen("/dev/urandom", "r");
    char rdata[13];
    fread(&rdata, 1, 12, urandom);
    rdata[12] = '\0';
    fclose(urandom);

    // convert bytes to integers
    uint32_t data1 = rdata[0] | ( (int)rdata[1] << 8 ) | ( (int)rdata[2] << 16 ) | ( (int)rdata[3] << 24 );
    uint32_t data2 = rdata[4] | ( (int)rdata[5] << 8 ) | ( (int)rdata[6] << 16 ) | ( (int)rdata[7] << 24 );
    uint32_t data3 = rdata[8] | ( (int)rdata[9] << 8 ) | ( (int)rdata[10] << 16 ) | ( (int)rdata[11] << 24 );

    //generate a random key
    uint32_t seed[] = { data1, data2, data3 };
    tfhe_random_generator_setSeed(seed,3);
    TFheGateBootstrappingSecretKeySet* key = new_random_gate_bootstrapping_secret_keyset(params);

    printf("Keyset generated!\n");
    printf("Exporting keys to files...\n");

    // export secret key to file
    FILE* secret_key = fopen("secret.key", "wb");
    export_tfheGateBootstrappingSecretKeySet_toFile(secret_key, key);
    fclose(secret_key);

    // export cloud key to file
    FILE* cloud_key = fopen("eval.key", "wb");
    export_tfheGateBootstrappingCloudKeySet_toFile(cloud_key, &key->cloud);
    fclose(cloud_key);

    printf("Finished exporting! Cleaning up...\n");

    delete_gate_bootstrapping_secret_keyset(key);
}

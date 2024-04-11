#include <tfhe/tfhe.h>
#include <tfhe/tfhe_io.h>
#include <tfhe/tfhe_garbage_collector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

TFheGateBootstrappingParameterSet* redsec_params_large(){
    int lwe_dim = 6144;
    int tlwe_dim = 8192;
    double std_dev = pow(2.,-48);
    double bootkey_std_dev = pow(2.,-46);
    double kskey_std_dev = pow(2., -41);


    LweParams* lwe_params = new_LweParams (lwe_dim, kskey_std_dev, std_dev);
    TLweParams* tlwe_params = new_TLweParams(tlwe_dim, 1, bootkey_std_dev, std_dev);
    TGswParams* tgsw_params = new_TGswParams(3, 10, tlwe_params);

    TfheGarbageCollector::register_param(lwe_params);
    TfheGarbageCollector::register_param(tlwe_params);
    TfheGarbageCollector::register_param(tgsw_params);

    return new TFheGateBootstrappingParameterSet(18, 1, lwe_params, tgsw_params);
}

TFheGateBootstrappingParameterSet* redsec_params_medium(){
    int lwe_dim = 3072;
    int tlwe_dim = 4096;
    double std_dev = pow(2.,-45);
    double bootkey_std_dev = pow(2.,-45);
    double kskey_std_dev = pow(2., -40);


    LweParams* lwe_params = new_LweParams (lwe_dim, kskey_std_dev, std_dev);
    TLweParams* tlwe_params = new_TLweParams(tlwe_dim, 1, bootkey_std_dev, std_dev);
    TGswParams* tgsw_params = new_TGswParams(3, 10, tlwe_params);

    TfheGarbageCollector::register_param(lwe_params);
    TfheGarbageCollector::register_param(tlwe_params);
    TfheGarbageCollector::register_param(tgsw_params);

    return new TFheGateBootstrappingParameterSet(18, 1, lwe_params, tgsw_params);
}

TFheGateBootstrappingParameterSet* redsec_params_small(){
    int lwe_dim = 500;
    int tlwe_dim = 1024;
    double std_dev = pow(2.,-11);
    double bootkey_std_dev = pow(2.,-36);
    double kskey_std_dev = pow(2.,-25);
    int32_t bootkey_basebits = 10;
    int32_t bootkey_length = 3;
    int32_t kskey_basebits = 1;
    int32_t ks_length = 18;

    LweParams* lwe_params = new_LweParams (lwe_dim, kskey_std_dev, std_dev);
    TLweParams* tlwe_params = new_TLweParams(tlwe_dim, 1, bootkey_std_dev, std_dev);
    TGswParams* tgsw_params = new_TGswParams(bootkey_length, bootkey_basebits, tlwe_params);

    TfheGarbageCollector::register_param(lwe_params);
    TfheGarbageCollector::register_param(tlwe_params);
    TfheGarbageCollector::register_param(tgsw_params);

    return new TFheGateBootstrappingParameterSet(ks_length, kskey_basebits, lwe_params, tgsw_params);
}

// NOTE: Set SECALPHA to 2^-15 to balance decreased LWE polynomial size
TFheGateBootstrappingParameterSet* redsec_params_small_v2(){
    int lwe_dim = 350;
    int tlwe_dim = 1024;
    int k = 1;
    double std_dev = pow(2.,-13);
    double bootkey_std_dev = pow(2.,-30);
    double kskey_std_dev = pow(2.,-25);
    int32_t bootkey_basebits = 3;
    int32_t bootkey_length = 10;
    int32_t kskey_basebits = 3;
    int32_t ks_length = 9;

    LweParams* lwe_params = new_LweParams(lwe_dim, kskey_std_dev, std_dev);
    TLweParams* tlwe_params = new_TLweParams(tlwe_dim, k, bootkey_std_dev, std_dev);
    TGswParams* tgsw_params = new_TGswParams(bootkey_length, bootkey_basebits, tlwe_params);

    TfheGarbageCollector::register_param(lwe_params);
    TfheGarbageCollector::register_param(tlwe_params);
    TfheGarbageCollector::register_param(tgsw_params);

    return new TFheGateBootstrappingParameterSet(ks_length, kskey_basebits, lwe_params, tgsw_params);
}


int main(int argc, char** argv) {

    // for wide networks, the medium and large parameters are better suited
    TFheGateBootstrappingParameterSet* params = redsec_params_small_v2();

    //generate a keypair
    uint32_t seed[] = { 0, 0, 0 };
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

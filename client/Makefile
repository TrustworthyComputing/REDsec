keygen: gen_secure_keyset.cpp ## Generate FHE keyset for CPU inference
	@g++ -o $@.out gen_secure_keyset.cpp -ltfhe-spqlios-fma
	@./$@.out -seclevel $(seclevel)

redcufhe-keygen: gen_secure_keyset_gpu.cu ## Generate FHE keyset for GPU inference
	@nvcc -o  $@.out gen_secure_keyset_gpu.cu -lredcufhe
	@./$@.out

encrypt-image: encrypt_image.cpp image_converter.py ## Encrypt CSV image for CPU inference
	@g++ -o encrypt.out encrypt_image.cpp -ltfhe-spqlios-fma
	@python3 image_converter.py --format $(format) --image $(image_path)
	@./encrypt.out image.ptxt

decrypt-image: decrypt_image.cpp ## Decrypt classification results from CPU inference
	@g++ -o decrypt.out decrypt_image.cpp -ltfhe-spqlios-fma
	@./decrypt.out $(format)

redcufhe-encrypt-image: encrypt_image_gpu.cu image_converter.py ## Encrypt CSV image for GPU inference
	@nvcc -o $@.out encrypt_image_gpu.cu -lredcufhe
	@python3 image_converter.py --format $(format) --image $(image_path)
	@./$@.out image.ptxt

redcufhe-decrypt-image: decrypt_image_gpu.cu ## Decrypt classification results from GPU inference
	@nvcc -o decrypt.out decrypt_image_gpu.cu -lredcufhe
	@./decrypt.out $(format)

encrypt-image-help: ## Print instructions for setting formatting options for encryption
	@echo "export 'format' (set to MNIST, CIFAR-10, or ImageNet)\ne.g. 'export format=MNIST'"
	@echo "export 'image_path' set to the path of the input image\ne.g. 'export image_path=0.csv'"
	@echo "run 'make encrypt-image' for TFHE or make 'redcufhe-encrypt-image' for REDcuFHE"

decrypt-image-help: ## Print instructions for setting formatting options for decryption
	@echo "export 'format' (set to MNIST, CIFAR-10, or ImageNet)\ne.g. 'export format=MNIST'"
	@echo "run 'make decrypt-image'"

clean: ## Delete executables, keys, and ciphertexts
	@rm *.out *.ptxt *.ctxt *.key 2>/dev/null || true

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
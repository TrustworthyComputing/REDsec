import pandas as pd
import sys
import os
import shutil

def empty_dict():
    params = dict()
    params['cwh'] = ""
    params['cww'] = ""
    params['csh'] = ""
    params['csw'] = ""
    params['csp'] = ""
    params['ctt'] = ""

    params['pwh'] = ""
    params['pww'] = ""
    params['psh'] = ""
    params['psw'] = ""
    params['psp'] = ""
    
    params['bns'] = ""
    params['bne'] = ""
    params['bbb'] = ""
    
    params['qsb'] = ""

    params['ver'] = ""

    return params

def print_params(paramnew, paramsold, outfile):
    #Convolution
    if paramnew['cwh'] != paramsold['cwh']:
        print("\tparams.conv.window.h = " + paramnew['cwh'], ";", file=outfile)
    if paramnew['cww'] != paramsold['cww']:
        print("\tparams.conv.window.w = " + paramnew['cww'], ";", file=outfile) 
    if paramnew['csh'] != paramsold['csh']:
        print("\tparams.conv.stride.h = " + paramnew['csh'], ";", file=outfile)
    if paramnew['csw'] != paramsold['csw']:
        print("\tparams.conv.stride.w = " + paramnew['csw'], ";", file=outfile)
    if paramnew['csp'] != paramsold['csp']:
        print("\tparams.conv.same_pad = " + paramnew['csp'], ";", file=outfile)
    if paramnew['ctt'] != paramsold['ctt']:
        print("\tparams.conv.tern_thresh = " + paramnew['ctt'], ";", file=outfile)
    
    #Pooling
    if paramnew['pwh'] != paramsold['pwh']:
        print("\tparams.pool.window.h = " + paramnew['pwh'], ";", file=outfile)
    if paramnew['pww'] != paramsold['pww']:
        print("\tparams.pool.window.w = " + paramnew['pww'], ";", file=outfile)
    if paramnew['psh'] != paramsold['psh']:
        print("\tparams.pool.stride.h = " + paramnew['psh'], ";", file=outfile)
    if paramnew['psw'] != paramsold['psw']:
        print("\tparams.pool.stride.w = " + paramnew['psw'], ";", file=outfile)
    if paramnew['psp'] != paramsold['psp']:
        print("\tparams.pool.same_pad = " + paramnew['psp'], ";", file=outfile)
    
    #BatchNorm
    if paramnew['bns'] != paramsold['bns']:
        print("\tparams.bnorm.use_scale = " + paramnew['bns'], ";", file=outfile)
    if paramnew['bne'] != paramsold['bne']:
        print("\tparams.bnorm.eps = " + paramnew['bne'], ";", file=outfile)
    if paramnew['bbb'] != paramsold['bbb']:
        print("\tparams.e_bias = " + paramnew['bbb'], ";", file=outfile)
    
    #Quantize
    if paramnew['qsb'] != paramsold['qsb']:
        print("\tparams.quant.shift_bits = " + paramnew['qsb'], ";", file=outfile)
    
    #version
    if paramnew['ver'] != paramsold['ver']:
        print("\tparams.version = " + paramnew['ver'], ";", file=outfile)


#start of code
if (len(sys.argv) < 3):
    print("Must Specify dataset and csv name\ne.g. python3 compiler.py mnist sample.csv")
    sys.exit(0)

if (".csv" not in sys.argv[2][-4:]):
    print("The network needs to be specified in CSV format")
    sys.exit(0)

#create net directory
invalid = ['.', '\\', '/', ',', '^', '<', '>', ':', '\"', '?', '*', '[', ']', ' ', '\t', '\n', '\r' ]
clean_netname = sys.argv[1]
for c in invalid:
	clean_netname = clean_netname.replace(c, "");
path = "../nets/"+clean_netname+"/"+sys.argv[2][:-4]
if(os.path.exists(path) == False):
   os.makedirs(path)
   print("Created new directory for network at ../nets/"+clean_netname + "/" + sys.argv[2][:-4])
else:
    print("Overwriting network files at ../nets/"+clean_netname+"/"+sys.argv[2][:-4])

flat = False
try:
    with open(sys.argv[2], newline='') as csvfile, \
        open("templates/tf_template.py", 'r') as tfinfile,\
        open(path+"/"+sys.argv[2][:-4]+".py", 'w') as tfoutfile, \
        open("templates/net_template.cpp", 'r') as cppinfile, \
        open(path+"/net.cpp", 'w') as cppoutfile, \
        open("templates/net_template.h", 'r') as hinfile, \
        open(path+"/net.h", 'w') as houtfile:
   
    #Preliminaries: Copy Template
        #REDsec
        red_line = 0
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        #LARQ Tensorflow
        tfin_line = 0
        for line in tfinfile:
            if "START" in line:
                print(line, file=tfoutfile, end='')
                break
            else:
                print(line, file=tfoutfile, end='')
            tfin_line += 1
    #End Preliminaries

            #Read file and extract model
        reader = pd.read_csv(csvfile, header=None).T
        
        params = empty_dict()
        params_old = empty_dict()
        layers = ["IntLayer"]
        for li in range(len(reader.T)):
            bNorm = False
            #Input, Convolution, Fully Connected (Index 0)
            if li == 0:
                if "input_size" in reader[0][0]:
                    instr = reader[0][0]
                    argstart = instr.find('(') + 1
                    argend = instr.find(')')
                    args = instr[argstart:argend].split(':')

                    #REDsec
                    params['conv_type'] = "E_NO_CONV" 
                    params['conv_dep'] = "SIZE_EMPTY"
                    print("\t  //Dimensions of input", file=cppoutfile)
                    print("\tlay_dim.hw.h = " +    args[0] + " ;", file=cppoutfile)
                    print("\tlay_dim.hw.w = " +    args[1] + " ;", file=cppoutfile)
                    print("\tlay_dim.in_dep = " +  args[2] + " ;", file=cppoutfile)
                    print("\tlay_dim.in_bits = " + args[3] + " ;", file=cppoutfile)
                    print("\tlay_dim.out_bits = SINGLE_BIT ;", file=cppoutfile)
                    print("\tlay_dim.filter_bits = SINGLE_BIT ; ;", file=cppoutfile)
                    print("\tlay_dim.bias_bits = SINGLE_BIT ;", file=cppoutfile)
                    print("\t//Start:TODO: Update to reflect preprocessing", file=cppoutfile)
                    print("\t  //up_bound is the range of pixel values ", file=cppoutfile)
                    print("\tlay_dim.up_bound = " + "2*" + str(2**int(args[3])-1) + " ;", file=cppoutfile)
                    print("\t  //scale is the mulitplicitive factor to map from", file=cppoutfile)
                    print("\t  //  floating point to integer", file=cppoutfile)
                    print("\tlay_dim.scale = " + str(2**int(args[3])-1) + " ;", file=cppoutfile)
                    print("\t//End:TODO: Update to reflect preprocessing ;", file=cppoutfile)
                    print("\n\n\t  //Model Params and Architechture", file=cppoutfile)

                    #LARQ Tensorflow
                    print("model.add(tf.keras.Input(", file=tfoutfile, end='')
                    print("("+ args[0] + "," + args[1] + "," + args[2] + ")", file=tfoutfile, end='')
                    print("))", file=tfoutfile)
                    act_str = "input_quantizer= lq.quantizers.NoOp(precision=" + args[3] + ")"
                else:
                    print("ERROR input_size: could not find input_size primitive")
                    exit
            elif "FullyConnect".lower() in str(reader[li][0]).lower():
                if not flat:
                    print("ERROR fully_connect: failed to flatten")
                    exit
                instr = reader[li][0]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')
                
                #REDsec
                params['conv_type'] = "E_FC" 
                params['cwh'] = params['cww'] = params['csh'] = params['csw'] = "1"
                params['conv_dep'] = args[0]
                params['ctt'] = args[1]

                #LARQ Tensorflow
                print("model.add(lq.layers.QuantDense(", file=tfoutfile, end='')
                print(args[0], ",", file=tfoutfile, end='')
                if float(args[1]) == 0:
                    print("kernel_quantizer=\"ste_sign\",", file=tfoutfile, end='')
                else:
                    print("kernel_quantizer=lq.quantizers.SteTern(threshold_value=" + args[1] + "),", file=tfoutfile, end='')
                print("kernel_constraint=\"weight_clip\", use_bias=False,", file=tfoutfile, end='')
                print(act_str, ",", file=tfoutfile, end='')
                print("))", file=tfoutfile)
            elif "Convolution" in reader[li][0]:
                instr = reader[li][0]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')
                #REDsec
                params['conv_type'] = "E_CONV"
                params['conv_dep'] = args[0]
                params['cwh'] = args[1][1:]
                params['cww'] = args[2][:-1]
                params['csh'] = args[3][1:]
                params['csw'] = args[4][:-1]
                params['csp'] = "true" if "same" in args[5].lower() else "false"
                params['ctt'] = args[6]

                #LARQ TensorFlow
                print("model.add(lq.layers.QuantConv2D(", file=tfoutfile, end='')
                print(args[0], ",", "("+ args[1][1:] + "," + args[2][:-1] +"),", file=tfoutfile, end='')
                print("strides=(" + args[3][1:] + "," + args[4][:-1] + "), padding=\""+ args[5].lower()+ "\",",  file=tfoutfile, end='')
                if float(args[6]) == 0:
                    print("kernel_quantizer=\"ste_sign\",", file=tfoutfile, end='')
                else:
                    print("kernel_quantizer=lq.quantizers.SteTern(threshold_value=" + args[6] + "),", file=tfoutfile, end='')
                print("kernel_constraint=\"weight_clip\", use_bias=False,", file=tfoutfile, end='')
                print(act_str, ",", file=tfoutfile, end='')
                print("))", file=tfoutfile)
            else:
                print("ERROR linear operation: expect convolution or fully connected layers")
            
            #Pooling (Index 1)
            if "MaxPool".lower() in str(reader[li][1]).lower():
                instr = reader[li][1]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')
                #REDsec
                params['pool_type'] = "E_MAXPOOL"
                params['pwh'] = args[0][1:]
                params['pww'] = args[1][:-1]
                params['psh'] = args[2][1:]
                params['psw'] = args[3][:-1]
                params['psp'] = "true" if "same" in args[4].lower() else "false"

                #LARQ TensorFlow
                print(len(args))
                print("model.add(tf.keras.layers.MaxPool2D(", file=tfoutfile, end='')
                print("("+ args[0][1:] + "," + args[1][:-1] +")" + ", strides=(" + \
                        args[2][1:] + "," + args[3][:-1] + "), padding=\"" + args[4].lower() +"\"", file=tfoutfile, end='')
                print("))", file=tfoutfile)
            elif "SumPool".lower() in str(reader[li][1]).lower():
                instr = reader[li][1]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')

                #REDsec
                params['pool_type'] = "E_SUMPOOL"
                params['pwh'] = args[0][1:]
                params['pww'] = args[1][:-1]
                params['psh'] = args[2][1:]
                params['psw'] = args[3][:-1]
                params['psp'] = "true" if "same" in args[4].lower() else "false"

                #LARQ Tensorflow
                print("model.add(tf.keras.layers.AveragePooling2D(", file=tfoutfile, end='')
                print("("+ args[0][1:] + "," + args[1][:-1] +")" + ", strides=(" + \
                        args[2][1:] + "," + args[3][:-1] + "), padding=\""+ args[4].lower() + "\"", file=tfoutfile, end='')
                print("))", file=tfoutfile)
            elif not pd.isnull(reader[li][1]):
                print("ERROR: unexpected command")
                print(li, 1, reader[li][1])
                exit
            else:
                params['pool_type'] = "E_NO_POOL"

            #BatchNorm (Index 2)
            if "BNorm".lower() in str(reader[li][2]).lower():
                instr = reader[li][2]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')
                bnorm = True

                #REDsec
                #momentum doesnt matter for inference
                params['bne'] = args[1]
                params['bns'] = "false"
                params['bbb'] = "E_BNORM"


                print("model.add(tf.keras.layers.BatchNormalization(", file=tfoutfile, end='')
                print("momentum=" + args[0] + ", epsilon=", args[1], file=tfoutfile, end='')
                print(", scale=False))", file=tfoutfile)
            elif not pd.isnull(reader[li][2]):
                print("ERROR: unexpected command")
                print(li, 2, reader[0][2])
                exit
            else:
                params['bbb'] = "E_NO_BIAS"
            
            #Dropout (Index 3)
            if "Dropout".lower() in str(reader[li][3]).lower():
                instr = reader[li][3]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')

                print("model.add(tf.keras.layers.Dropout(", args[0], "))", file=tfoutfile)
            elif not pd.isnull(reader[li][3]):
                print("ERROR: unexpected command")
                print(li, 3, reader[li][3])
                exit
            
            #Activation (Index 4)
            if "Sign".lower() in str(reader[li][4]).lower():
                #REDsec
                layers.append("BinLayer")
                params['act_type'] = "E_ACTIVATION_SIGN"
                #LARQ Tensorflow
                act_str = "input_quantizer= \"ste_sign\""
            elif "ReLU".lower() in str(reader[li][4]).lower():
                instr = reader[li][4]
                argstart = instr.find('(') + 1
                argend = instr.find(')')
                args = instr[argstart:argend].split(':')

                #REDsec
                layers.append("IntLayer")
                params['act_type'] = "E_ACTIVATION_RELU"
                params['qsb'] = args[0]
                #LARQ Tensorflow
                act_str = ""
                act_str += "input_quantizer= lq.quantizers.DoReFa(k_bit=" + args[0] + ")"
            elif bNorm:
                print("ERROR bnorm: BNorm requires an activation function")
                print(li, 4)
                exit
            else:
                layers.append("IntLayer")
                params['act_type'] = "E_ACTIVATION_NONE"

            #Flatten (Index 5)
            if "Flatten".lower() in str(reader[li][5]).lower():
                print("model.add(tf.keras.layers.Flatten())", file=tfoutfile)
                flat =  True

            #print REDsec
            print("\n\t\t//Layer " + str(len(layers)-2), file=cppoutfile)
            print_params(params, params_old, cppoutfile) ;
            print("\tconv_dep = ", params['conv_dep'], ";", file = cppoutfile)
            print("#if defined(GPU_ENC)\n\tlayer" + str(len(layers)-2) + " = new " + layers[-2] + "(" + params['conv_type'] + ", conv_dep, " + params['pool_type'] + \
                  ", " + params['act_type'] + ", &params);", file=cppoutfile)
            print("#else\n\tlayer" + str(len(layers)-2) + " = new " + layers[-2]+"("+ params['conv_type']+ ", conv_dep, " + params['pool_type']+ \
                    ", "+  params['act_type']+ ", &params, bk) ;\n#endif", file=cppoutfile)
            params_old = params.copy()

        print("model.add(tf.keras.layers.Activation(\"softmax\"))", file=tfoutfile)
        for line in tfinfile:
            print(line, file=tfoutfile, end='')
            tfin_line += 1

    #Start REDsec cpp code
            #copy template, the write prep code
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        print("\tp_dim = &lay_dim ;", file=cppoutfile)
        for i in range(len(layers)-1):
           print("\tp_dim = (tDimensions*) layer" + str(i) + "->prep(in_file, p_dim) ;", file=cppoutfile)

            #copy template, the write execute code
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        prev_str = "in_data"
        prev_str_2 = "in_data"
        for i in range(len(layers)-1):
            if "Bin".lower() in layers[i+1].lower():
                print("#if !defined(GPU_ENC)\n\tbdata = (tBit*)", file=cppoutfile, end='')
                sv_str = "(tBit*) bdata"
            else:
                print("#if !defined(GPU_ENC)\n\tmbdata = (tMultiBit*)", file=cppoutfile, end = '')
                sv_str = "(tFixedPoint*) mbdata"
            print("layer" + str(i) + "->execute(" + prev_str +") ;", file=cppoutfile)
            if "Bin".lower() in layers[i+1].lower():
                print("#else\n\tbdata = (tBitPacked*)", file=cppoutfile, end='')
                sv_str_2 = "(tBitPacked*) bdata"
            else:
                print("#else\n\tmbdata = (tMultiBitPacked*)", file=cppoutfile, end = '')
                sv_str_2= "(tMultiBitPacked*) mbdata"
            print("layer" + str(i) + "->execute(" + prev_str_2 +") ;", file=cppoutfile)
            print("#endif", file=cppoutfile)
            prev_str = sv_str
            prev_str_2 = sv_str_2
            
            #copy template, the write weight code
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        for i in range(len(layers)-1):
            #if "Bin".lower() in layers[i+1].lower():
            #    print("\tbdata = (tBit*)", file=cppoutfile, end='')
            #else:
            #    print("\tmbdata = (tMultiBit*)", file=cppoutfile, end = '')
            print("layer" + str(i) + "->export_weights(out_file) ;", file=cppoutfile)

            #copy template, the write helper function code
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        print("\t\tmemcpy(out, &(layer"+str(len(layers)-2)+"->out_dim), sizeof(*out)) ;", file=cppoutfile)


            #finish copying template
        for line in cppinfile:
            if "START" in line:
                print(line, file=cppoutfile, end='')
                break
            else:
                print(line, file=cppoutfile, end='')
            red_line += 1
        #Finished REDsec cpp code

        #Start net.h generation
        h_line = 0
        for line in hinfile:
            if "START" in line:
                print(line, file=houtfile, end='')
                break
            else:
                print(line, file=houtfile, end='')
            h_line += 1
        for i in range(len(layers)-1):
            print("\t\t"+layers[i]+"* layer"+ str(i) + " ;", file=houtfile)
        for line in hinfile:
            if "START" in line:
                print(line, file=houtfile, end='')
                break
            else:
                print(line, file=houtfile, end='')
            h_line += 1
        #Finish generating h file

except IOError:
    print("Error opening file")
    print("e.g. python3 compiler.py mnist sample.csv")
    print("Ensure ./template folder is properly populated")

# Copy template files to net directory
shutil.copyfile("templates/main.cpp", path+"/main.cpp")
shutil.copyfile("templates/Makefile", path+"/Makefile")
shutil.copyfile("templates/weight_convert.cpp", path+"/weight_convert.cpp")
#copy over mnist data
shutil.copyfile("templates/mnist.h", path+"/../mnist.h")
shutil.copyfile("templates/mnist_data.csv", path+"/../mnist_data.csv")

# Create CUDA copies of net.cpp/net.h and main.cpp
shutil.copyfile(path+"/net.cpp", path+"/net.cu")
shutil.copyfile(path+"/net.h", path+"/net.cuh")
shutil.copyfile(path+"/main.cpp", path+"/main.cu")

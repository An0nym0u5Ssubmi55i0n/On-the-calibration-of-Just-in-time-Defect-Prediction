import argparse
import os
from padding import padding_data
import pickle
import numpy as np 
from evaluation import evaluation_model
from train import train_model
from utils import _read_tsv
from tokenization_of_bert import tokenization_for_codebert

def read_args():
    parser = argparse.ArgumentParser()


 

     # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  
    parser.add_argument('-valid', action='store_true')
    parser.add_argument('-train_data', type=str, help='the directory of our training data')   
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')    

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=100, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=4, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=120, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=768, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=3, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='codebert4jit_msg_code', help='where to save the snapshot')    

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no_cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()

    if params.cp is True:
        number_cp_datapoints = 1000
        number_reps = 100
        # for reproducibility
        np.random.seed(1)
        random_seeds = np.random.randint(100000, size=number_reps)
        params.q_hats = None
        print(f"[Random Seeds] {random_seeds}")

        print("\n")
        for iteration in range(99,100,1):
            print(f"[START ITERATION] Iteration = {iteration}")
            current_model_path = params.save_dir + f"/iteration_{iteration}/epoch_final_step_final.pt"
            # delete trained models from previous iteration
            if iteration > 0:
                current_model_path = params.save_dir.replace(str(iteration-1), str(iteration)) + f"/epoch_final_step_final.pt"
                previous_model_path = params.save_dir + f"/epoch_final_step_final.pt"
                print(f"Current Model Path iteration {iteration} = {current_model_path} and previous one = {previous_model_path}")
                # if os.path.isfile(previous_model_path):
                #     os.remove(previous_model_path)
                # else:
                #    raise RuntimeError(f"Previous Model could not be deleted. in iteration {iteration} for file {previous_model_path}")
            

            # load data
            print(f"[LOAD DATA]")
            dictionary = pickle.load(open(params.dictionary_data, 'rb'))
            dict_msg, dict_code = dictionary

            train_data = pickle.load(open(params.train_data, 'rb'))
            test_data = pickle.load(open(params.pred_data, 'rb'))
            train_ids, train_labels, train_msgs, train_codes = train_data
            test_ids, test_labels, test_msgs, test_codes = test_data
            number_train_datapoints = len(train_ids)
            number_test_datapoints = len(test_ids)
            combined_ids = train_ids + test_ids
            combined_labels = train_labels + test_labels
            combined_msgs = train_msgs + test_msgs
            combined_codes = train_codes + test_codes
            
            # shuffle dataset randomly
            print(f"[SHUFFLE DATA]")
            np.random.seed(random_seeds[iteration])
            np.random.shuffle(combined_ids)
            np.random.seed(random_seeds[iteration])
            np.random.shuffle(combined_labels)
            np.random.seed(random_seeds[iteration])
            np.random.shuffle(combined_msgs)
            np.random.seed(random_seeds[iteration])
            np.random.shuffle(combined_codes)

            # split shuffled dataset (calibration data is taken from training set)
            print(f"[SPLIT DATA]")
            shuffled_cp_data = (combined_ids[:number_cp_datapoints], combined_labels[:number_cp_datapoints], combined_msgs[:number_cp_datapoints], combined_codes[:number_cp_datapoints])
            shuffled_train_data = (combined_ids[number_cp_datapoints:number_train_datapoints], combined_labels[number_cp_datapoints:number_train_datapoints], combined_msgs[number_cp_datapoints:number_train_datapoints], combined_codes[number_cp_datapoints:number_train_datapoints])
            shuffled_test_data = (combined_ids[number_train_datapoints:], combined_labels[number_train_datapoints:], combined_msgs[number_train_datapoints:], combined_codes[number_train_datapoints:])

            assert len(shuffled_cp_data[0]) == number_cp_datapoints
            assert len(shuffled_cp_data[1]) == number_cp_datapoints
            assert len(shuffled_cp_data[2]) == number_cp_datapoints
            assert len(shuffled_cp_data[3]) == number_cp_datapoints

            assert len(shuffled_train_data[0]) == number_train_datapoints-number_cp_datapoints
            assert len(shuffled_train_data[1]) == number_train_datapoints-number_cp_datapoints
            assert len(shuffled_train_data[2]) == number_train_datapoints-number_cp_datapoints
            assert len(shuffled_train_data[3]) == number_train_datapoints-number_cp_datapoints

            assert len(shuffled_test_data[0]) == number_test_datapoints
            assert len(shuffled_test_data[1]) == number_test_datapoints
            assert len(shuffled_test_data[2]) == number_test_datapoints
            assert len(shuffled_test_data[3]) == number_test_datapoints

            
            # prepare training 
            print("[PREPARE TRAIN MODEL]")
            print(f"len(codes) = {len(shuffled_train_data[3])}, len(ids) = {len(shuffled_train_data[0])}")
            print(f"params = {params}")
            # tokenize the code and msg
            train_pad_msg = tokenization_for_codebert(data=shuffled_train_data[2], max_length= params.msg_length, flag ='msg')
            train_pad_code = tokenization_for_codebert(data=shuffled_train_data[3], max_length= params.code_length, flag ='code')
            final_train_data = (train_pad_msg, train_pad_code, np.array(shuffled_train_data[1]), dict_msg, dict_code)
            print(np.shape(train_pad_msg), np.shape(train_pad_code))

            # training
            print("[TRAIN MODEL - SKIPPED]")
            # train_model(data=final_train_data, params=params, iteration=iteration)        
                
            # TODO 
            # replicate og results
            
            # preapre calibration 
            # print("[PREPARE CALIBRATION MODEL]")
            # params.calibrate_model = True
            params.load_model = current_model_path
            # # tokenize the code and msg
            # cp_pad_msg = tokenization_for_codebert(data=shuffled_cp_data[2], max_length= params.msg_length,  flag ='msg')
            # cp_pad_code = tokenization_for_codebert(data=shuffled_cp_data[3], max_length= params.code_length, flag ='code')
            # final_cp_data = (cp_pad_msg, cp_pad_code, np.array(shuffled_cp_data[1]), dict_msg, dict_code)
            # print(np.shape(cp_pad_msg), np.shape(cp_pad_code))

            # # calibration
            # print("[CALIBRATE MODEL/APPLY CP]")
            # evaluation_model(data=final_cp_data, params=params, iteration=iteration)

            # preapre test 
            print("[PREPARE TEST MODEL]")
            params.calibrate_model = False
            # tokenize the code and msg
            test_pad_msg = tokenization_for_codebert(data=shuffled_test_data[2], max_length= params.msg_length,  flag ='msg')
            test_pad_code = tokenization_for_codebert(data=shuffled_test_data[3], max_length= params.code_length, flag ='code')
            final_test_data = (test_pad_msg, test_pad_code, np.array(shuffled_test_data[1]), dict_msg, dict_code)
            print(np.shape(test_pad_msg), np.shape(test_pad_code))
            # load currently trained model
            params.calibrate_model = False

            # testing
            print("[TEST MODEL]")
            evaluation_model(data=final_test_data, params=params, iteration=iteration)

        print("[FINISHED]")
        exit(0)
    if params.train is True:

        ## read dict data
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary
        
        '''
        ## read tsv data
        labels = list()
        msgs = list()
        codes = list()
        lines = _read_tsv(params.train_data)
        for line in lines:
            labels.append(line[0])
            codes.append(line[1])
            msgs.append(line[2])
        
        '''
        ## read pickle data
        data = pickle.load(open(params.train_data, 'rb'))
        
        ids, labels, msgs, codes = data
        data_len = len(ids)
        print(data_len)
        if params.valid != True:
            ids = ids[0:int(data_len*0.9)]
            labels = labels[0:int(data_len*0.9)]
            codes = codes[0:int(data_len*0.9)]
            msgs =  msgs[0:int(data_len*0.9)]
        else:
            ids = ids[int(data_len*0.9):]
            labels = labels[int(data_len*0.9):]
            codes = codes[int(data_len*0.9):]
            msgs =  msgs[int(data_len*0.9):]

        print(len(codes), len(ids))
        # tokenize the code and msg
        pad_msg = tokenization_for_codebert(data=msgs, max_length= params.msg_length, flag ='msg')
        pad_code = tokenization_for_codebert(data=codes, max_length= params.code_length, flag ='code')
        data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        print(np.shape(pad_msg), np.shape(pad_code))
        # training
        if params.valid != True:
            train_model(data=data, params=params)        
        else:
            evaluation_model(data=data, params=params)

    elif params.predict is True:
        
        ## read dict data
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        dict_msg, dict_code = dictionary
        '''
        ## for tsv data
        labels = list()
        msgs = list()
        codes = list()
        lines = _read_tsv(params.pred_data)
        for line in lines:
            labels.append(line[0])
            codes.append(line[1])
            msgs.append(line[2])
        '''
        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data
        
        # tokenize the code and msg
        pad_msg = tokenization_for_codebert(data=msgs, max_length= params.msg_length,  flag ='msg')
        pad_code = tokenization_for_codebert(data=codes, max_length= params.code_length, flag ='code')
        data = (pad_msg, pad_code, np.array(labels), dict_msg, dict_code)
        print(np.shape(pad_msg), np.shape(pad_code))
        # testing
        evaluation_model(data=data, params=params)
    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()

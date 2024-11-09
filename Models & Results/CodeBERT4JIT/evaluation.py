from model import  CodeBERT4JIT
from calculate_metrics import evaluate_calibration
from utils import mini_batches, pad_input_matrix
from sklearn.metrics import roc_auc_score    
from sklearn import metrics
import torch 
from tqdm import tqdm
import numpy as np


def evaluation_model(data, params, iteration=0):
    
    # preprocess on the code and msg data
    pad_msg, pad_code, labels,  dict_msg, dict_code= data
    pad_code_input_ids, pad_code_input_masks, pad_code_segment_ids = pad_code
    pad_msg_input_ids, pad_msg_input_masks, pad_msg_segment_ids = pad_msg
    
    pad_msg_input_ids = np.array(pad_msg_input_ids)
    pad_msg_input_masks = np.array(pad_msg_input_masks)
    pad_msg_segment_ids = np.array(pad_msg_segment_ids)
    
    
    pad_input_matrix(pad_code_input_ids, params.code_line)
    pad_input_matrix(pad_code_input_masks, params.code_line)
    pad_input_matrix(pad_code_segment_ids, params.code_line)
    
    pad_code_input_ids = np.array(pad_code_input_ids)
    pad_code_input_masks = np.array(pad_code_input_masks)
    pad_code_segment_ids = np.array(pad_code_segment_ids)
    
    # build batches 
    batches = mini_batches(X_msg_input_ids=pad_msg_input_ids, X_msg_masks=pad_msg_input_masks, X_msg_segment_ids= pad_msg_segment_ids, X_code_input_ids =pad_code_input_ids, X_code_masks=pad_code_input_masks, X_code_segment_ids=pad_code_segment_ids, Y=labels, mini_batch_size=params.batch_size)
    
    # set up parameters
    
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    print(f"Params = {params}")
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    # TODO del again
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model =  CodeBERT4JIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    # model.load_state_dict(torch.load(params.load_model))
    model.load_state_dict(torch.load('../../trained-qt-model/iteration_99/epoch_final_step_final.pt'))

    ## ---------------------- Evalaution Process ---------------------------- ##
    model.eval()  # eval mode 
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            
            msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = batch
            if torch.cuda.is_available():                
                msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id, labels = torch.tensor(msg_input_id).cuda(),torch.tensor(msg_input_mask).cuda(),torch.tensor(msg_segment_id).cuda(), torch.tensor(code_input_id).cuda(),torch.tensor(code_input_mask).cuda(),torch.tensor(code_segment_id).cuda(), torch.cuda.FloatTensor(labels.astype(int))
                
            else:                
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                
                predict = model.forward(msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id)
                # print(f"[Model 3] predict = {predict} and detach = {predict.cpu().detach()} and numpy = {predict.cpu().detach().numpy()} and list = {predict.cpu().detach().numpy().tolist()}")
                predict = predict.cpu().detach().numpy().tolist()
            else:
               
                predict = model.forward(msg_input_id, msg_input_mask, msg_segment_id, code_input_id, code_input_mask, code_segment_id)
                # print(f"[Model 3] predict = {predict} and detach = {predict.detach()} and numpy = {predict.detach().numpy()} and list = {predict.detach().numpy().tolist()}")
                predict = predict.detach().numpy().tolist()

            # print(f"[Model 4] predict {predict} and allpredrict = {all_predict}")
            # print(f"[Model 5] Labels =  {labels}")
            all_predict += predict
            all_label += labels.tolist()
    
    # for index, ele in enumerate(all_predict):
    #     print(f"Label = {all_label[index]} and predict = {ele}\n")
    # compute the AUC scores
    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)
    if params.cp and False:
        q_hats = evaluate_calibration(all_predict, all_label, f"iteration_{iteration}_evaluation_qt.csv", params.calibrate_model, params.q_hats)
        params.q_hats = q_hats

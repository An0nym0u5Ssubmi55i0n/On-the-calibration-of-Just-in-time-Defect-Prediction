#!/bin/sh
# testing openstack
# printf "python main.py -predict -pred_data '../../openstack_test_changed.pkl'  -load_model '../../op_pretrained_model.pt' -dictionary_data '../../op_dict.pkl'"
# python main.py -predict -pred_data '../../openstack_test_changed.pkl'  -load_model '../../op_pretrained_model.pt' -dictionary_data '../../op_dict.pkl'

# testing qt
# printf "python main.py -predict -pred_data '../../qt_test_changed.pkl'  -load_model '../../qt_pretrained_model.pt' -dictionary_data '../../qt_dict.pkl'"
# python main.py -predict -pred_data '../../qt_test_changed.pkl'  -load_model '../../qt_pretrained_model.pt' -dictionary_data '../../qt_dict.pkl'

# training
# printf "python main.py -train -train_data '../../openstack_train_changed.pkl' -save-dir '../../trained_op_model'  -dictionary_data '../../op_dict.pkl'"
# 
# python main.py -train_data '../../openstack_train_changed.pkl' -save-dir '../../trained_op_model'  -dictionary_data '../../op_dict.pkl'
# 

# cp
# openstack
# printf "python main.py -cp -train_data '../../openstack_train_changed.pkl' -pred_data '../../openstack_test_changed.pkl' -save-dir '../../trained-op-model'  -dictionary_data '../../op_dict.pkl'"
# 
# python main.py -cp -train_data '../../openstack_train_changed.pkl' -pred_data '../../openstack_test_changed.pkl' -save-dir '../../trained-op-model'  -dictionary_data '../../op_dict.pkl'

# qt
printf "python main_replication.py -cp -train_data '../../qt_train_changed.pkl' -pred_data '../../qt_test_changed.pkl' -save-dir '../../trained-qt-model'  -dictionary_data '../../qt_dict.pkl'"

python main_replication.py -cp -train_data '../../qt_train_changed.pkl' -pred_data '../../qt_test_changed.pkl' -save-dir '../../trained-qt-model'  -dictionary_data '../../qt_dict.pkl'





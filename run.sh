# General flow: tx_train.yaml -> tx_ft -> tx_val -> tx_test
###
 # @Description: 
 # @Version: 
 # @Author: Xuanying Chen
 # @Date: 2022-03-09 09:34:42
 # @LastEditTime: 2022-03-09 09:39:04
### 

# tx_train: trains the model.
# tx_ft: uses data-replay to address forgetting. (refer Sec 4.4 in paper)
# tx_val: learns the weibull distribution parameters from a kept aside validation set.
# tx_test: evaluate the final model
# x above can be {1, 2, 3, 4}

# NB: Please edit the paths accordingly.
# NB: Please change the batch-size and learning rate if you are not running on 8 GPUs.
# (if you find something wrong in this, please raise an issue on GitHub)

# Task 1
python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52125' --resume --config-file ./configs/OWOD/t1/t1_train.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t1"

# No need to finetune in Task 1, as there is no incremental component.

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t1/t1_val.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "../drive/MyDrive/output/t1_final" MODEL.WEIGHTS "../drive/MyDrive/output/t1/model_final.pth"

python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t1/t1_test.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.005 OUTPUT_DIR "../drive/MyDrive/output/t1_final" MODEL.WEIGHTS "../drive/MyDrive/output/t1/model_final.pth"


# Task 2
cp -r ../drive/MyDrive/output/t1 ../drive/MyDrive/output/t2

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t2/t2_train.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t2" MODEL.WEIGHTS "../drive/MyDrive/output/t2/model_final.pth"

cp -r ../drive/MyDrive/output/t2 ../drive/MyDrive/output/t2_ft

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t2/t2_ft.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t2_ft" MODEL.WEIGHTS "../drive/MyDrive/output/t2_ft/model_final.pth"

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t2/t2_val.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "../drive/MyDrive/output/t2_final" MODEL.WEIGHTS "../drive/MyDrive/output/t2_ft/model_final.pth"

python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t2/t2_test.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.005 OUTPUT_DIR "../drive/MyDrive/output/t2_final" MODEL.WEIGHTS "../drive/MyDrive/output/t2_ft/model_final.pth"


# Task 3
cp -r ../drive/MyDrive/output/t2_ft ../drive/MyDrive/output/t3

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t3/t3_train.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t3" MODEL.WEIGHTS "../drive/MyDrive/output/t3/model_final.pth"

cp -r ../drive/MyDrive/output/t3 ../drive/MyDrive/output/t3_ft

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t3/t3_ft.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t3_ft" MODEL.WEIGHTS "../drive/MyDrive/output/t3_ft/model_final.pth"

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52133' --config-file ./configs/OWOD/t3/t3_val.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OWOD.TEMPERATURE 1.5 OUTPUT_DIR "../drive/MyDrive/output/t3_final" MODEL.WEIGHTS "../drive/MyDrive/output/t3_ft/model_final.pth"

python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t3/t3_test.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.005 OUTPUT_DIR "../drive/MyDrive/output/t3_final" MODEL.WEIGHTS "../drive/MyDrive/output/t3_ft/model_final.pth"


# Task 4
cp -r ../drive/MyDrive/output/t3_ft ../drive/MyDrive/output/t4

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52127' --resume --config-file ./configs/OWOD/t4/t4_train.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t4" MODEL.WEIGHTS "../drive/MyDrive/output/t4/model_final.pth"

cp -r ../drive/MyDrive/output/t4 ../drive/MyDrive/output/t4_ft

python tools/train_net.py --num-gpus 1 --dist-url='tcp://127.0.0.1:52126' --resume --config-file ./configs/OWOD/t4/t4_ft.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00125 OUTPUT_DIR "../drive/MyDrive/output/t4_ft" MODEL.WEIGHTS "../drive/MyDrive/output/t4_ft/model_final.pth"

python tools/train_net.py --num-gpus 1 --eval-only --config-file ./configs/OWOD/t4/t4_test.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.005 OUTPUT_DIR "../drive/MyDrive/output/t4_final" MODEL.WEIGHTS "../drive/MyDrive/output/t4_ft/model_final.pth"
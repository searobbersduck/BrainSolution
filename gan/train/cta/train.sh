UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10001  ddp_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200805.py

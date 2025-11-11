generate_mvtec:
	python generate_corrupted_mvtec.py

train_mvtec_DINL:
	python train_mvtec_DINL.py

inference_mvtec_ATTA:
	python inference_mvtec_ATTA.py

train_infer_RD4AD: train_mvtec_RD4AD inference_mvtec_RD4AD

train_mvtec_RD4AD:
	python train_mvtec_RD4AD.py --gpu 0

inference_mvtec_RD4AD:
	python inference_mvtec_RD4AD.py
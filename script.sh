conda create -n ison_env python=3.9
conda activate ison_env
pip install -r requirement
python inference.py --ckpt ParkingNet_mbv3_small.onnx --sub_ckpt1 LPRNet.onnx --sub_ckpt2 sgie_car_maker.onnx --inference_method Stream --data_cfg dyiot_trafficNet_lpr_car_maker_v1.0.0_onnx.json

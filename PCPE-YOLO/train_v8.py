import sys
import argparse
import os

# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    results = model.train(#data='/root/autodl-tmp/KITTI/traindata.yaml',  # 训练参数均可以重新设置
                        data='/root/autodl-tmp/visdrone_yolo/traindata.yaml',
                        #data="D:/Data/visdrone_yolo/traindata.yaml",
                        #data='/root/autodl-tmp/NWPU_yolov8/data.yaml',
                        epochs=200,
                        imgsz=640, 
                        workers=8, 
                        batch=16,
                        pretrained=False,
                        #amp=False,
                        #resume=True,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/root/autodl-tmp/.autodl/PIG-YOLO/IG-YOLO/ultralytics-main/runs/detect/train24/weights/last.pt', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

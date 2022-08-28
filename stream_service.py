import multiprocessing
import queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Queue

import cv2
import torch
from fastapi import FastAPI
from streamlink import Streamlink
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import set_logging, check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

opt = {
    "weights": "runs/train/exp/weights/epoch_069.pt",
    "weights2": "runs/train/exp/weights/epoch_098.pt",
    # ruta de los pesos de los modelos resultado de ajuste
    "img-size": 640,  # tamaño de la imágen a la que se transforma
    "conf-thres": 0.25,  # threshold para la inferencia.
    "iou-thres": 0.45,  # NMS IoU threshold para inferencia
    "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": [],  # clases que se desean filtrar, como tenemos 2 (casco y no casco las dos serán usadas)
    "video-stream": {  # streams de cámaras del cuál se realiza la inferencia
        "pl": "https://www.youtube.com/watch?v=zu6yUYEERwA",
        "ec": "https://www.youtube.com/watch?v=EnXaKSxnrqc",
        "ph": "https://www.youtube.com/watch?v=qgNTbBn0JCY&ab_channel=TheRealSamuiWebcam"
    }
}


def init_steam(url: str, sl: Streamlink, my_queue: Queue):
    print("Iniciando stream %s", url)
    # se define que live requerido y en la calidad deseada, esto afecta al rendimiento según los fps
    video_stream = sl.streams(url)["720p"].url
    # usamos la libreria de open cv u cv2 para procesar el video entrante en funcion de frames
    street_stream = cv2.VideoCapture(video_stream)

    # determinamos los parametros para generar los sublicps de videos que se generan cuando detecta y no detecta casco
    fps = street_stream.get(cv2.CAP_PROP_FPS)
    w = int(street_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(street_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pt = 0
    with torch.no_grad():  # permite que la gpu no guarde en cache el cálculo del gradiente
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else ["casco", "sin casco"]
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        frames_video = 200
        grabar_video = False
        output = None
    while street_stream.isOpened():

        ret, img0 = street_stream.read()

        if ret:
            img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=[0, 1],
                                   agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    if conf >= 0.7:

                        clase = names[int(cls)]
                        if not grabar_video:
                            date = datetime.now()
                            v_name = "%s%s%s%s" % (date.day, date.hour, date.minute, date.second)
                            output = cv2.VideoWriter(
                                f'/video/v-{v_name}.mp4',
                                cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
                            print(f"--- Class Detection --- [{clase}] -- recording")
                            grabar_video = True

                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                if output and output.isOpened():
                    output.write(img0)
                    frames_video = frames_video - 1
                    grabar_frame = False

                    if frames_video == 0:
                        frames_video = 200
                        grabar_video = False
                        output.release()
            else:
                grabar_frame = True

        if output and output.isOpened() and grabar_frame:
            output.write(img0)
            frames_video = frames_video - 1

            if frames_video == 0:
                frames_video = 200
                grabar_video = False
                output.release()
        try:
            item = my_queue.get_nowait()
        except queue.Empty:
            continue
        else:
            if item:
                print("Almacenando Video")
                break
    street_stream.release()
    output.release()


def init_threads() -> [ThreadPoolExecutor, multiprocessing.Queue, Streamlink, FastAPI]:
    return ThreadPoolExecutor(max_workers=1), multiprocessing.Queue(), Streamlink(), FastAPI()

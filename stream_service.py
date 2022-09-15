import os
import queue
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Queue

import cv2
import numpy as np
import psycopg2
import pytz
import torch
from fastapi import FastAPI
from numpy import random
from streamlink import Streamlink

from models.experimental import attempt_load
from send_bucket_s3 import upload_to_aws, SOURCE_FILE
from utils.datasets import letterbox
from utils.general import set_logging, check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

threads, source_queue, source_sl, app, conn, timezone = ThreadPoolExecutor(max_workers=3), \
                                                        Queue(), \
                                                        Streamlink(), \
                                                        FastAPI(), \
                                                        psycopg2.connect(
                                                            database='gkzejkjy',
                                                            user='gkzejkjy',
                                                            password='dIOoqb7HmpxybbKsB1SeWQmZnCNVhOa9',
                                                            host='chunee.db.elephantsql.com',
                                                            port='5432',
                                                        ), \
                                                        pytz.timezone("America/Guayaquil")

opt = {
    "weights": "runs/train/exp/weights/epoch_069.pt",
    "weights2": "runs/train/exp/weights/epoch_098.pt",
    # ruta de los pesos de los modelos resultado de ajuste
    "img-size": 640,  # tamaño de la imágen a la que se transforma
    "conf-thres": 0.25,  # threshold para la inferencia.
    "iou-thres": 0.45,  # NMS IoU threshold para inferencia
    "device": 'cpu',  # device to run our model i.e. 0 for GPU or 0,1,2,3 or cpu
    "classes": [],  # clases que se desean filtrar, como tenemos 2 (casco y no casco las dos serán usadas)
    "video-stream": { # streams de cámaras del cuál se realiza la inferencia
        "pl": "https://www.youtube.com/watch?v=zu6yUYEERwA",
        "ec": "https://www.youtube.com/watch?v=EnXaKSxnrqc",
        "ph": "https://www.youtube.com/watch?v=qgNTbBn0JCY&ab_channel=TheRealSamuiWebcam"
    },
    'query_insert': 'INSERT INTO INFRACCIONES (fecha, hora, num_infracciones, num_no_infracciones, camara, name_camera) '
                    'VALUES (%s, %s, %s, %s, %s, %s)',
    'query_select': 'SELECT HORA, NUM_INFRACCIONES, NUM_NO_INFRACCIONES '
                    'FROM INFRACCIONES WHERE FECHA = %s AND CAMARA = %s',
    'query_all_select': 'SELECT FECHA, CAMARA, NAME_CAMERA FROM INFRACCIONES',
    'query_one_camera_select': 'SELECT NAME_CAMERA FROM INFRACCIONES WHERE CAMARA = %s',
}


def init_steam(url: str, sl: Streamlink, my_queue: Queue, is_rtsp, url_almacenada, camera_name):
    print("Iniciando stream %s" % url)
    speed = 1.2
    num_infracciones, num_no_infracciones = 0, 0
    fecha_actual = datetime.now(timezone)
    start_time = datetime.strptime(fecha_actual.time().strftime('%H:%M'), '%H:%M')
    dia_actual = fecha_actual.strftime('%y-%m-%d')  # ya es st
    saved_camera = fetch_camera_data(url_almacenada)
    if not saved_camera:
        save_dates(dia_actual, 0, 0, 0, url_almacenada, camera_name)
    else:
        camera_name = saved_camera[0]
        print("Stream ya existente", url_almacenada, camera_name)

    init_frames_video = 250
    # se define que live requerido y en la calidad deseada, esto afecta al rendimiento según los fps
    if not is_rtsp:
        video_stream = sl.streams(url)["720p"].url
    else:
        init_frames_video = 180
        speed = 2.3
        video_stream = url
    # usamos la libreria de open cv u cv2 para procesar el video entrante en funcion de frames
    street_stream = cv2.VideoCapture(video_stream)

    # determinamos los parametros para generar los sublicps de videos que se generan cuando detecta y no detecta casco
    fps = street_stream.get(cv2.CAP_PROP_FPS)
    w = int(street_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(street_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, w, h)

    pt = 0
    with torch.no_grad():  # permite que la gpu no guarde en cache el cálculo del gradiente
        weights, imgsz = opt['weights2'], opt['img-size']
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

        frames_video = init_frames_video
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
                            if int(c) == 1:
                                i = int(n)
                            else:
                                ni = int(n)

                        for *xyxy, conf, cls in reversed(det):
                            if conf >= 0.7:

                                clase = names[int(cls)]
                                if not grabar_video:
                                    date = datetime.now()
                                    v_name = "%s%s%s%s.mp4" % (date.day, date.hour, date.minute, date.second)
                                    output = cv2.VideoWriter(
                                        f'{SOURCE_FILE}/{v_name}',
                                        cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
                                    print(f"--- Class Detection --- [{clase}] -- recording")
                                    print("Grabando video %s" % v_name)
                                    num_infracciones = num_infracciones + i
                                    num_no_infracciones = num_no_infracciones + ni
                                    print("Capturando infractores", num_infracciones, "No infractores",
                                          num_no_infracciones)
                                    grabar_video = True
                                    # cv2_imshow(img0)

                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                        if output and output.isOpened():
                            output.write(img0)
                            frames_video = frames_video - 1
                            grabar_frame = False

                            if frames_video == 0:
                                frames_video = init_frames_video

                                grabar_video = False
                                output.release()
                                threads.submit(upload_to_aws, v_name, f'{url_almacenada}_s{v_name}', speed)

                    else:
                        grabar_frame = True

                if output and output.isOpened() and grabar_frame:
                    output.write(img0)
                    frames_video = frames_video - 1

                    if frames_video == 0:
                        frames_video = init_frames_video

                        grabar_video = False
                        output.release()
                        threads.submit(upload_to_aws, f'{v_name}', f'{url_almacenada}_s{v_name}', speed)

            try:
                item = my_queue.get_nowait()
            except queue.Empty:

                end_time = datetime.strptime(datetime.now(timezone).time().strftime('%H:%M'), '%H:%M')

                if str(end_time - start_time)[:4] == '0:03':  # se envia cada 6 min
                    print("Enviando numero de infractores a la bd", dia_actual, num_infracciones, num_no_infracciones,
                          url_almacenada, url)
                    save_dates(dia_actual, start_time.hour, num_infracciones, num_no_infracciones, url_almacenada,
                               camera_name)
                    num_infracciones = 0
                    num_no_infracciones = 0
                    start_time = datetime.strptime(datetime.now(timezone).time().strftime('%H:%M'), '%H:%M')
                continue

            else:
                if item:
                    print("Enviando numero de infractores a la bd", dia_actual, num_infracciones, num_no_infracciones,
                          url_almacenada, url)
                    save_dates(dia_actual, start_time.hour, num_infracciones, num_no_infracciones, url_almacenada,
                               camera_name)
                    print("Saliendo del Stream", url)
                    break

        print("Video eliminado")
        street_stream.release()
        output.release()
        os.remove(f'{SOURCE_FILE}/{v_name}')


def validate_rtsp(stream_url):
  if re.search(r'rtsp:\/\/[a-z0-9]{3,8}:[a-z0-9]{3,8}@([0-9]+.){4}:[0-9]{3,4}([-a-zA-Z0-9@:%_\+.~#?&//=]*)', stream_url):
    return True
  elif re.search(r'[\/\/a-zA-Z0-9@:%._\+~#=]{2,256}\.\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', stream_url):
    return False
  else:
    return -2


def save_dates(dia_actual, start_time, num_infracciones, num_no_infracciones, camara, camera_name):
    cur = conn.cursor()
    data = opt['query_insert'] % (
        f"'{dia_actual}'", start_time, num_infracciones, num_no_infracciones, f"'{camara}'", f"'{camera_name}'")
    cur.execute(data)
    conn.commit()
    print("Guardando data")


def fetch_camera_data(url):
    cur = conn.cursor()
    data = opt['query_one_camera_select'] % f"'{url}'"
    cur.execute(data)
    return cur.fetchone()


def fetch_all_cameras():
    cur = conn.cursor()
    data = opt['query_all_select']
    cur.execute(data)
    arr = []
    for i in cur.fetchall():
        my_data = {'ip': i[1], 'name': i[2]}
        if my_data not in arr:
            arr.append(my_data)
    return arr


def fetch_data(fecha: str, url) -> dict:
    cur = conn.cursor()
    data = opt['query_select'] % (f"'{fecha}'", f"'{url}'")
    cur.execute(data)

    my_dict = {}

    for i in cur.fetchall():
        if not i[0] in my_dict.keys():
            my_dict[i[0]] = {
                'num_infracciones': 0,
                'num_no_infracciones': 0
            }
        sub_dict = my_dict[i[0]]
        sub_dict["num_infracciones"] = sub_dict["num_infracciones"] + i[1]
        sub_dict["num_no_infracciones"] = sub_dict["num_no_infracciones"] + i[2]
        my_dict[i[0]] = sub_dict

    return my_dict


def fetch_all_data() -> list:
    cur = conn.cursor()
    data = opt['query_all_select']
    cur.execute(data)
    arr = []
    for i in cur.fetchall():
        if {i[1], i[2]} not in arr:
            arr.append({i[1], i[2]})
    return arr


if __name__ == '__main__':
    conn = psycopg2.connect(
        database='gkzejkjy',
        user='gkzejkjy',
        password='dIOoqb7HmpxybbKsB1SeWQmZnCNVhOa9',
        host='chunee.db.elephantsql.com',
        port='5432',
    )
    cur = conn.cursor()
    query = 'SELECT CAMARA FROM INFRACCIONES WHERE CAMARA = %s'
    data = query % "'45.186.5.4430:554'"
    cur.execute(data)
    if cur.fetchone():
        print("SI TODO NICE")
    conn.close()
    # print(rows)
    # my_data = {
    # }
    # my_data["hora"] = 1
    # my_data["a"] = 1
    # my_data["v"] = 1
    # print(my_data)
    # init_steam('http://173.255.219.215/stream/helmet-cam/index.m3u8', Streamlink(), None, True)
    # print(validate_rtsp("rtsp://grupo9:grupo9@45.186.5.30:554/stream1"))

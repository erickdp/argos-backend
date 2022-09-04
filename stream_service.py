import multiprocessing
import os
import queue
import re
import psycopg2
import pytz
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Queue

import cv2
import numpy as np
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
                                                        multiprocessing.Queue(), \
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
    "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": [],  # clases que se desean filtrar, como tenemos 2 (casco y no casco las dos serán usadas)
    "video-stream": {  # streams de cámaras del cuál se realiza la inferencia
        "pl": "https://www.youtube.com/watch?v=zu6yUYEERwA",
        "ec": "https://www.youtube.com/watch?v=EnXaKSxnrqc",
        "ph": "https://www.youtube.com/watch?v=qgNTbBn0JCY&ab_channel=TheRealSamuiWebcam"
    },
    'query_insert': 'INSERT INTO INFRACCIONES (fecha, hora, num_infracciones) VALUES (%s, %s, %s)',
    'query_select': 'SELECT HORA, NUM_INFRACCIONES FROM INFRACCIONES WHERE FECHA = %s',
}


def init_steam(url: str, sl: Streamlink, my_queue: Queue, is_rtsp):
    speed = 1.2
    num_infracciones = 0
    fecha_actual = datetime.now(timezone)
    start_time = datetime.strptime(fecha_actual.time().strftime('%H:%M'), '%H:%M')
    dia_actual = fecha_actual.strftime('%y-%m-%d')  # ya es str
    init_frames_video = 250
    print("Iniciando stream %s" % url)
    # se define que live requerido y en la calidad deseada, esto afecta al rendimiento según los fps
    if not is_rtsp:
        video_stream = sl.streams(url)["720p"].url
    else:
        init_frames_video = 200
        speed = 2.3
        video_stream = url
    # usamos la libreria de open cv u cv2 para procesar el video entrante en funcion de frames
    street_stream = cv2.VideoCapture(video_stream)

    # determinamos los parametros para generar los sublicps de videos que se generan cuando detecta y no detecta casco
    fps = street_stream.get(cv2.CAP_PROP_FPS)
    w = int(street_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(street_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, w, h)

    # descomentar para ver live en vivo
    # while True:
    #     ret, frame = street_stream.read()
    #     if ret:
    #         cv2.imshow('Frame', frame)
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    # street_stream.release()
    # cv2.destroyAllWindows()

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

                        for *xyxy, conf, cls in reversed(det):
                            if conf >= 0.7:

                                clase = names[int(cls)]
                                if not grabar_video:
                                    date = datetime.now()
                                    v_name = "%s%s%s%s.mp4" % (date.day, date.hour, date.minute, date.second)
                                    output = cv2.VideoWriter(
                                        f'{SOURCE_FILE}/{v_name}',
                                        cv2.VideoWriter_fourcc(*'MP4V'), 30, (w, h))
                                    print(f"--- Class Detection --- [{clase}] -- recording")
                                    print("Grabando video %s" % v_name)
                                    num_infracciones = num_infracciones + 1
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
                                num_infracciones = 0
                                grabar_video = False
                                output.release()
                                threads.submit(upload_to_aws,
                                               f'{v_name}',
                                               f's{v_name}',
                                               speed)

                    else:
                        grabar_frame = True

                if output and output.isOpened() and grabar_frame:
                    output.write(img0)
                    frames_video = frames_video - 1

                    if frames_video == 0:
                        frames_video = init_frames_video
                        num_infracciones = 0
                        grabar_video = False
                        output.release()
                        threads.submit(upload_to_aws, f'{v_name}',
                                       f's{v_name}', speed)

            try:
                item = my_queue.get_nowait()
            except queue.Empty:
                continue
            else:
                if item:
                    conn.close()
                    print("Saliendo del Stream y cerrando conexion")
                    break

            end_time = datetime.strptime(datetime.now(timezone).time().strftime('%H:%M'), '%H:%M')

            if str(end_time - start_time)[:4] == '0:05':
                save_dates(dia_actual, start_time.hour, num_infracciones)
                print("Enviando numero de infractores a la bd", dia_actual, num_infracciones)
                start_time = datetime.strptime(datetime.now(timezone).time().strftime('%H:%M'), '%H:%M')
                continue

        # print("Video eliminado")
        street_stream.release()
        output.release()
        os.remove(f'{SOURCE_FILE}/{v_name}')


def validate_rtsp(stream_url):
    if re.search(r'rtsp:\/\/[a-z0-9]{3,8}:[a-z0-9]{3,8}@([0-9]+.){4}:[0-9]{3,4}([-a-zA-Z0-9@:%_\+.~#?&//=]*)',
                 stream_url):
        return True
    elif re.search(r'[\/\/a-zA-Z0-9@:%._\+~#=]{2,256}\.\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', stream_url):
        return False
    else:
        return -2


def save_dates(dia_actual, start_time, num_infracciones):

    cur = conn.cursor()
    data = opt['query_insert'] % (f"'{dia_actual}'", start_time, num_infracciones)
    cur.execute(data)
    conn.commit()

    # timezone = pytz.timezone("America/Guayaquil")
    # temp = 1
    # while temp != 0:
    #     cur = conn.cursor()
    #
    #     fecha_actual = datetime.now(timezone)
    #     hora_actual = fecha_actual.time().strftime('%H:%M')
    #     dia_actual = fecha_actual.strftime('%y-%m-%d')  # ya es str
    #
    #     print(fecha_actual, hora_actual, dia_actual, type(dia_actual))
    #
    #     start_time = datetime.strptime(hora_actual, '%H:%M')
    #     # sleep(5)
    #     end_time = datetime.strptime(datetime.now(timezone).time().strftime('%H:%M'), '%H:%M')
    #
    #     end_time = datetime.strptime('01:33', '%H:%M')
    #
    #     final_time = end_time - start_time
    #     print(type(final_time), final_time, str(final_time)[:4] == '0:01', '%s' % '2022-09-03')
    #
    #     # query = "INSERT INTO INFRACCIONES (fecha, hora, num_infracciones) VALUES (%s, %s, %s)"
    #     data = opt['query'] % (f"'{dia_actual}'", start_time.hour, 5)
    #     print(data)
    #     cur.execute(data)
    #
    #     conn.commit()
    #     temp = temp - 1
    # conn.close()
    # conn.close()


def fetch_data(fecha: str) -> list:
    cur = conn.cursor()
    query = 'SELECT HORA, NUM_INFRACCIONES FROM INFRACCIONES WHERE FECHA = %s'
    data = query % f"'{fecha}'"
    cur.execute(data)
    return cur.fetchall()


if __name__ == '__main__':
    conn = psycopg2.connect(
        database='gkzejkjy',
        user='gkzejkjy',
        password='dIOoqb7HmpxybbKsB1SeWQmZnCNVhOa9',
        host='chunee.db.elephantsql.com',
        port='5432',
    )
    cur = conn.cursor()
    query = 'SELECT HORA, NUM_INFRACCIONES FROM INFRACCIONES WHERE FECHA = %s'
    data = query % "'22-09-04'"
    cur.execute(data)
    rows = cur.fetchall()
    print(rows)
    # init_steam('http://173.255.219.215/stream/helmet-cam/index.m3u8', Streamlink(), None, True)
    # print(validate_rtsp("rtsp://grupo9:grupo9@45.186.5.30:554/stream1"))

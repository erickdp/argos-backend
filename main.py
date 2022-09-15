from typing import Union

from fastapi import HTTPException

from stream_service import init_steam, threads, source_queue, source_sl, app, validate_rtsp, fetch_data, fetch_all_data, \
    fetch_all_cameras
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# single_thread, my_queue, sl, app = init_threads()

@app.get("/", status_code=202)
async def root():
    return {"msg": "RESTful de IA. Helmet-Detection"}

@app.get("/stream", status_code=200)
async def root(
        stream_url: Union[str, None] = None,
        stream_save: Union[str, None] = None,
        camera_name: Union[str, None] = None
):
    type_url = validate_rtsp(stream_url)
    if type_url == -2:
        raise HTTPException(status_code=404, detail="Recurso incorrecto")
    threads.submit(init_steam, stream_url, source_sl, source_queue, type_url, stream_save, camera_name)
    return {"msg": "Iniciando procesado de stream"}

@app.get("/stop", status_code=202)
async def say_hello():
    source_queue.put(True)
    return {"msg": "Deteniendo procesamiento de stream"}

@app.get("/fetch", status_code=200)
async def get_dates(date: Union[str, None] = None, camara: Union[str, None] = None):
    dates = fetch_data(date, camara)
    return {
        "msg": "exito",
        'data': dates
    }

@app.get("/fetchAll", status_code=200)
async def get_all_dates():
    dates = fetch_all_data()
    return {
        "msg": "exito",
        'data': dates
    }


@app.get("/fetchAllCameras", status_code=200)
async def get_all_cameras():
    cameras = fetch_all_cameras()
    return {
        "msg": "exito",
        'data': cameras
    }

# if __name__ == '__main__':
#     uvicorn.run(app, port=8080, host='0.0.0.0')

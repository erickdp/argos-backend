from typing import Union

from fastapi import HTTPException

from stream_service import init_steam, threads, source_queue, source_sl, app, validate_url


# single_thread, my_queue, sl, app = init_threads()

@app.get("/")
async def root():
    return {"msg": "RESTful de IA. Helmet-Detection"}


@app.get("/stream", status_code=200)
async def root(stream_url: Union[str, None] = None):
    url = validate_url(stream_url)
    if not url:
        raise HTTPException(status_code=404, detail="Recurso incorrecto")
    threads.submit(init_steam, url.string, source_sl, source_queue)
    return {"msg": "Iniciando procesado de stream"}


@app.get("/stop", status_code=200)
def say_hello():
    source_queue.put(True)
    return {"msg": "Deteniendo procesamiento de stream"}

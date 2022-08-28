from typing import Union

from stream_service import init_steam, threads, source_queue, source_sl, app


# single_thread, my_queue, sl, app = init_threads()


@app.get("/stream")
async def root(stream_url: Union[str, None] = None):
    threads.submit(init_steam, stream_url, source_sl, source_queue)


@app.get("/stop")
def say_hello():
    source_queue.put(True)

import queue
from time import sleep
from typing import Union

from stream_service import init_threads, init_steam

single_thread, my_queue, sl, app = init_threads()


@app.get("/stream")
async def root(stream_url: Union[str, None] = None):
    single_thread.submit(init_steam, stream_url, sl, my_queue)


@app.get("/stop")
def say_hello():
    my_queue.put(True)

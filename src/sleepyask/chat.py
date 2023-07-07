import os
import time
import queue
import json
import asyncio
import aiohttp
import aiofiles


from openai_async import openai_async

# TODO: Shouldn't be that hard to extend to other OpenAI functions but chat is the only one I'm familiar with atm
class Sleepyask:
    """
    This class provides functions which use ayncio to ask multiple questions to ChatGPT simultaneously.
    This allows users to aggregate a large number of responses from ChatGPT.
    """

    def __init__(self, 
                 configs : dict ={}, 
                 rate_limit: int = 5, 
                 api_key: str = "", 
                 timeout: int = 100, 
                 verbose: bool = False, 
                 retry_time: int = 60,
                 system_text: str = ""
    ):
        """
        `configs` a dict containing parameters which will be part of the payload such as model, temperature, etc
        `rate_limit` the maximum number of questions you would like to ask a minute.
        `api_key` OpenAI API key
        `timeout` the amount of time to wait before timing out
        `out_path` the path in which the responses will be outputted
        """
        self.configs = configs
        self.rate_limit = rate_limit
        self.api_key = api_key
        self.timeout = timeout
        self.verbose = verbose
        self.retry_time = retry_time
        self.system_text = system_text

    async def get_asked_ids(self, out_path):
        asked_ids = set()

        if os.path.isfile(out_path):
            async with aiofiles.open(out_path, "r") as infile:
                async for row in infile:
                    try:
                        row = json.loads(row)
                        if "question_id" in row:
                            asked_ids.add(str(row["question_id"]))
                    except:
                        pass

        return asked_ids

    async def start(self, question_list, out_path):
        self.out_path = out_path
        self.succeed = 0
        self.file_lock = asyncio.Lock()
        self.question_queue = asyncio.Queue()

        asked_ids = await self.get_asked_ids(out_path)

        for question in question_list:
            if str(question["id"]) not in asked_ids:
                await self.question_queue.put(question)

        await asyncio.gather(*[self.async_ask() for _ in range(self.rate_limit)])


    async def log(self, response):
        async with self.file_lock:
            async with aiofiles.open(self.out_path, "a") as outfile:
                await outfile.write(json.dumps(response))
                await outfile.write("\n")

            self.succeed += 1

    async def async_ask(self, question_list):
        while self.succeed < len(question_list):
            question = await self.question_queue.get()
            question_index = question["id"]
            question_text = question["text"]

            try:
                if self.verbose:
                    print(f"[sleepyask] INFO | ID {question_index} | ASKING: {question_text}")

                payload = {
                    "messages": [
                        {"role": "system", "content": self.system_text},
                        {"role": "user", "content": question_text}
                    ],
                    **self.configs
                }
                response = await openai_async.chat_complete(payload=payload, api_key=self.api_key, timeout=self.timeout)

                if self.verbose:
                    print(f"[sleepyask] INFO | ID {question_index} | RECEIVED: {response.text}")

                if response.status_code != 200:
                    if self.verbose:
                        print(f"[sleepyask] INFO | ID {question_index} | {response.status_code}")
                    raise ValueError("Should be 200")

                await self.log({"question_id": question_index, **json.loads(response.text), "question": question_text, **self.configs})
            except:
                if self.verbose:
                    print(f"[sleepyask] INFO | ID {question_index} | ERROR")
                await self.question_queue.put(question)

            await asyncio.sleep(self.retry_time)
            
        await asyncio.sleep(0)

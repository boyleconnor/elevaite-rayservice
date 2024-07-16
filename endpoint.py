import os

import numpy as np
from ray import serve

# These imports are used only for type hints:
from starlette.requests import Request
from transformers import pipeline


def numpy_to_std(obj):
    """Convert all objects in dict (recursively) from numpy types to vanilla
    Python types."""
    if isinstance(obj, list):
        new_obj = []
        for item in obj:
            new_obj.append(numpy_to_std(item))
        return new_obj
    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if type(key) is not str:
                raise TypeError(
                    f"Dictionary contains invalid key {key!r}; {type(key)=}"
                )
            new_obj[key] = numpy_to_std(value)
        return new_obj
    elif type(obj) in (int, float, str):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        raise TypeError(f"Could not serialize evaluation object: {obj}")


@serve.deployment(num_replicas=2)
class ModelDeployment:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "dslim/bert-base-NER")
        self.pipe = pipeline(model=model_name)

    async def __call__(self, request: Request) -> dict:
        request_json = await request.json()
        args = request_json["args"]
        kwargs = request_json["kwargs"]
        return numpy_to_std(self.pipe(*args, **kwargs))

deployment_graph = ModelDeployment.bind()

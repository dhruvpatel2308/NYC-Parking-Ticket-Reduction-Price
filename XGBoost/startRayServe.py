import ray
from ray import serve
ray.init(address='auto', namespace='serve')
serve.start(detached=True)
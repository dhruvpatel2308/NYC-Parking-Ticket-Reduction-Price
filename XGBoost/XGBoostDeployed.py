import pickle
import json
import ray
from ray import serve

# @serve.deployment(num_replicas=2, route_prefix="/regressor")
class XGB:
	def __init__(self):
		with open('model.pkl', 'r') as f:
			self.model = pickle.load(f)

		print("Model Loaded")

	async def __call__(self, starlette_request):
		payload = await starlette_request.json()

		print("Worked received a request: ", payload)

		input_vector = [
			payload["pregnancies"],
			payload["glucose"],
			payload["blood_pressure"],
			payload["skin_thickness"],
			payload["insulin"],
			payload["bmi"],
			payload["diabetes_pedigree"],
			payload["age"]
		]

		prediction = self.model.predict([input_vector])[0]
		return {"result": prediction}


# # Start the server
# serve.start(detached=True)
# XGB.deploy()

model = XGB()
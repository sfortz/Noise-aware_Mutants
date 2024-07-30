from qiskit_ibm_runtime import QiskitRuntimeService
 
service = QiskitRuntimeService()

print(service.backends())

print(service.backend("ibm_kyoto"))
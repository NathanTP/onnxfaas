import json
import subprocess as sp
import pathlib
import sys
from . import kv
import signal

# Allow importing models from sibling directory (python nonsense) 
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
import models.ferplus as ferplus
import models.bertsquad as bertsquad 

class InvocationError():
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return msg

modelsDir = (pathlib.Path(__file__).parent.parent / "models").resolve()

class LocalModel:
    def __init__(self, modelName, objStore, provider="CUDA_ExecutionProvider"):
        self.objStore = objStore

        if modelName == "ferplus":
            self.model = ferplus.Model(provider=provider)
        elif modelName == "bertsquad":
            self.model = bertsquad.Model(provider=provider)
        else:
            raise RuntimeError("Unrecognized model name: ", modelName)


    def pre(self, name, inputKey=None):
        if inputKey is None:
            inputKey = name+".in"

        inputs = self.objStore.get(inputKey)
        ret = self.model.pre(inputs)
        self.objStore.put(name+".pre", ret)
        return name+".pre"
    

    def run(self, name):
        inputs = self.objStore.get(name+".pre")
        ret = self.model.run(inputs)
        self.objStore.put(name+".run", ret)
        return name+".run"


    def post(self, name):
        inputs = self.objStore.get(name+".run")
        ret = self.model.post(inputs)
        self.objStore.put(name+".final", ret)
        return name+".final"


    def inputs(self, name):
        inputs = self.model.inputs()
        self.objStore.put(name+".in", inputs)
        return name+".in"

    def close(self):
        pass

class RemoteModel:
    def __init__(self, modelName, objStore, provider="CUDA_ExecutionProvider"):
        """Create a new model executor for modelName. Arguments will be passed through objStore."""
        self.objStore = objStore
        self.provider = provider

        if modelName == "ferplus":
            modelPath = modelsDir / "ferplus" / "ferplus.py"
        else:
            modelPath = modelsDir / "bertsquad" / "bertsquad.py"

        self.proc = sp.Popen(["python3", str(modelPath)], bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, text=True)

        # Note: local models would wait until everything was ready before
        # returning. For remote, that's all hapening in a different process so
        # it can overlap with local computation. The remote init time will show
        # up as a slower first invocation of a function. I'm not sure how best
        # to report this.


    def _invoke(self, arg):
        self.proc.stdin.write(json.dumps(arg) + "\n")
        rawResp = self.proc.stdout.readline()
        resp = json.loads(rawResp)
        if resp['error'] is not None:
            raise InvocationError(resp['error'])


    def pre(self, name, inputKey=None):
        if inputKey is None:
            inputKey = name+".in"

        req = {
            "func" : "pre",
            "provider" : self.provider,
            "inputKey" : inputKey,
            "outputKey" : name+".pre"
        }

        self._invoke(req)
        return name+".pre"


    def run(self, name):
        req = {
            "func" : "run",
            "provider" : self.provider,
            "inputKey" : name+".pre",
            "outputKey" : name+".run"
        }

        self._invoke(req)
        return name+".run"


    def post(self, name):
        req = {
            "func" : "post",
            "provider" : self.provider,
            "inputKey" : name+".run",
            "outputKey" : name+".final"
        }

        self._invoke(req)
        return name+".final"

    def inputs(self, name):
        req = {
            "func" : "inputs",
            "provider" : self.provider,
            "inputKey" : None,
            "outputKey" : name+".in"
        }

        self._invoke(req)
        return name+".in"

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


argFields = [
        "func", # Function to invoke (either "pre", "run", or "post")
        "provider", # onnxruntime provider 
        "inputKey", # Key name to read from for input
        "outputKey", # Key name that outputs should be written to
        ]

def remoteServer(modelClass):
    def onExit(sig, frame):
        print("Function executor exiting")
        sys.exit(0)

    signal.signal(signal.SIGINT, onExit)

    # Eagerly set up any needed state. It's not clear how realistic this is,
    # might switch it around later.
    modelClass.imports()
    modelStates = {
            "CPUExecutionProvider" : modelClass(provider="CPUExecutionProvider"),
            "CUDAExecutionProvider" : modelClass(provider="CUDAExecutionProvider")
            }

    # objStore = fakefaas.kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)
    objStore = kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)

    for rawCmd in sys.stdin:
        try:
            cmd = json.loads(rawCmd)
        except json.decoder.JSONDecodeError as e:
            err = "Failed to parse command (must be valid JSON): " + str(e)
            print(json.dumps({ "error" : err }), flush=True)
            continue

        for k in argFields:
            if k not in cmd:
                err = "missing required argument " + str(k)
                print(json.dumps({ "error" : err }), flush=True)
                continue

        curModel = modelStates[cmd['provider']]


        if cmd['func'] == "pre":
            funcInputs = objStore.get(cmd['inputKey'])
            funcOut = curModel.pre(funcInputs)
        elif cmd['func'] == 'run':
            funcInputs = objStore.get(cmd['inputKey'])
            funcOut = curModel.run(funcInputs)
        elif cmd['func'] == 'post':
            funcInputs = objStore.get(cmd['inputKey'])
            funcOut = curModel.post(funcInputs)
        elif cmd['func'] == 'inputs':
            funcOut = curModel.inputs()
        else:
            err = "unrecognized function " + str(cmd['func'])
            print(json.dumps({ "error" : err }), flush=True)
            continue

        objStore.put(cmd['outputKey'], funcOut)

        print(json.dumps({"error" : None}), flush=True)

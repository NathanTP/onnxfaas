import cProfile
from contextlib import contextmanager
import models.ferplus as ferplus
import models.bertsquad as bertsquad 
import time
import argparse
import redis
import pickle
import fakefaas
import fakefaas.kv

class prof:
    def __init__(self):
        self.total = 0.0
        self.ncall = 0

    def increment(self, n):
        self.total += n
        self.ncall += 1

    def total(self):
        return self.total

    def mean(self):
        return self.total / self.ncall

# ms
timeScale = 1000

@contextmanager
def timer(name, timers):
   start = time.time() * timeScale 
   try:
       yield
   finally:
       if name not in timers:
           timers[name] = prof()
       timers[name].increment((time.time()*timeScale) - start)


def runTest(state, objStore, profTimes, niter=1):
    inputs = state.inputs()

    for repeat in range(niter):
        for i, testIn in enumerate(inputs):
            iterID = "{}.{}".format(repeat, i)
            with timer("pre", profTimes):
                processed = state.pre(testIn)
                objStore.put(iterID+".pre", processed)

            with timer("run", profTimes):
                desProcessed = objStore.get(iterID+".pre")

                modelOut = state.run(desProcessed)

                objStore.put(iterID+".run", modelOut)

            with timer("post", profTimes):
                desModelOut = objStore.get(iterID+".run")

                finalOut = state.post(modelOut)

                objStore.put(iterID+".final", finalOut)

            objStore.delete(iterID+".pre", iterID+".run", iterID+".final")
        

def main(model, niter=1, use_gpu=True, serialize=False):
    if serialize:
        objStore = fakefaas.kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)
    else:
        objStore = fakefaas.kv.Local(copyObjs=False, serialize=False)

    times = {}
    
    with timer("imports", times):
        model.imports()

    with timer("init", times):
        if use_gpu:
            state = model(provider="CUDAExecutionProvider") 
        else:
            state = model(provider="CPUExecutionProvider") 

    with timer("run", times):
        runTest(state, objStore, times, niter=niter)

    # prof_file = state.session.end_profiling()
    # print("onnxruntime profile at: ", prof_file)

    print("Times (ms): ")
    print({ name : v.mean() for name, v in times.items() })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for faas-like onnxruntime benchmarking")
    parser.add_argument("-c", "--cpu", action='store_true',
            help="use CPU execution provider (rather than the default CUDA provider)")
    parser.add_argument("-n", "--niter", type=int, default=1, help="Number of test iterations to perform")
    parser.add_argument("-m", "--model", type=str, default="ferplus", help="Which model to run, either 'bertsquad' or 'ferplus'")
    parser.add_argument("-s", "--serialize", action='store_true', help="Serialize data (and store in kv store) in between steps. You must have a redis server running for this.")

    args = parser.parse_args()

    if args.model == "ferplus":
        model = ferplus.Model
    elif args.model == "bertsquad":
        model = bertsquad.Model

    main(model, niter=args.niter, use_gpu = not args.cpu, serialize=args.serialize)

    # Cprofile
    # profFile = "fer10kGPU.prof"
    # cProfile.runctx("main(model, niter=args.niter, use_gpu = not args.cpu, serialize=args.serialize)", filename=profFile)
    # print("Profile at: ", profFile)

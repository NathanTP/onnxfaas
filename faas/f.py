import cProfile
from contextlib import contextmanager
import models.ferplus as ferplus
import models.bertsquad as bertsquad 
import time
import argparse
import redis
import pickle

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


def runTest(state, niter=1, profTimes=None, serialize=False):
    inputs = state.inputs()

    kv = redis.Redis(password="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy")

    for repeat in range(niter):
        for i, testIn in enumerate(inputs):
            iterID = "{}.{}".format(repeat, i)
            if profTimes is not None:
                with timer("pre", profTimes):
                    processed = state.pre(testIn)
                    if serialize:
                        kv.set(iterID+".pre", pickle.dumps(processed))

                with timer("run", profTimes):
                    if serialize:
                        desProcessed = pickle.loads(kv.get(iterID+".pre"))
                    else:
                        desProcessed = processed

                    modelOut = state.run(desProcessed)

                    if serialize:
                        kv.set(iterID+".run", pickle.dumps(modelOut))

                with timer("post", profTimes):
                    if serialize:
                        desModelOut = pickle.loads(kv.get(iterID+".run"))
                    else:
                        desModelOut = modelOut

                    finalOut = state.post(modelOut)

                    kv.set(iterID+".final", pickle.dumps(finalOut))

                kv.delete(iterID+".pre", iterID+".run", iterID+".final")
            else:
                processed = state.pre(testIn)
                modelOut = state.run(processed)
                finalOut = state.post(modelOut)


def main(model, profile=False, niter=1, use_gpu=True, serialize=False):
    if profile:
        times = {}
        
        with timer("imports", times):
            model.imports()

        with timer("init", times):
            if use_gpu:
                state = model(profile=profile, provider="CUDAExecutionProvider") 
            else:
                state = model(profile=profile, provider="CPUExecutionProvider") 

        with timer("run", times):
            runTest(state, niter=niter, profTimes=times, serialize=serialize)

        prof_file = state.session.end_profiling()
        print("onnxruntime profile at: ", prof_file)
        print("Times (ms): ")
        print({ name : v.mean() for name, v in times.items() })
    else:
        model.imports()
        if use_gpu:
            state = model(profile=profile, provider="CUDAExecutionProvider") 
        else:
            state = model(profile=profile, provider="CPUExecutionProvider") 
        runTest(state, niter=niter, serialize=serialize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for faas-like onnxruntime benchmarking")
    parser.add_argument("-c", "--cpu", action='store_true',
            help="use CPU execution provider (rather than the default CUDA provider)")
    parser.add_argument("-p", "--profile", action='store_true', help="Enable self-profiling")
    parser.add_argument("-n", "--niter", type=int, default=1, help="Number of test iterations to perform")
    parser.add_argument("-m", "--model", type=str, default="ferplus", help="Which model to run, either 'bertsquad' or 'ferplus'")
    parser.add_argument("-s", "--serialize", action='store_true', help="Serialize data (and store in kv store) in between steps. You must have a redis server running for this.")

    args = parser.parse_args()

    if args.model == "ferplus":
        model = ferplus.Model
    elif args.model == "bertsquad":
        model = bertsquad.Model

    main(model, profile=args.profile, niter=args.niter, use_gpu = not args.cpu, serialize=args.serialize)

    # Cprofile
    # profFile = "fer10kGPU.prof"
    # cProfile.runctx("main(profile=False, niter=niter)", globals=globals(), locals=locals(), sort="cumulative", filename=profFile)
    # print("Profile at: ", profFile)

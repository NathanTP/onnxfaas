import cProfile
from contextlib import contextmanager
import models.ferplus as ferplus
import models.bertsquad as bertsquad 
import time
import argparse

@contextmanager
def timer(name, timers):
   start = time.time()
   try:
       yield
   finally:
       if name not in timers:
           timers[name] = 0.0
       timers[name] += time.time() - start


def runTest(state, niter=1):
    inputs = state.inputs()

    for repeat in range(niter):
        for testIn in inputs:
            processed = state.pre(testIn)

            # predict using the deployed model
            modelOut = state.run(processed)

            finalOut = state.post(modelOut)


def main(model, profile=False, niter=1, use_gpu=True):
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
            runTest(state, niter=niter)

        prof_file = state.session.end_profiling()
        print("onnxruntime profile at: ", prof_file)
        print("Times: ")
        print(times)
    else:
        model.imports()
        if use_gpu:
            state = model(profile=profile, provider="CUDAExecutionProvider") 
        else:
            state = model(profile=profile, provider="CPUExecutionProvider") 
        runTest(state, niter=niter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for faas-like onnxruntime benchmarking")
    parser.add_argument("-c", "--cpu", action='store_true',
            help="use CPU execution provider (rather than the default CUDA provider)")
    parser.add_argument("-p", "--profile", action='store_true', help="Enable self-profiling")
    parser.add_argument("-n", "--niter", type=int, default=1, help="Number of test iterations to perform")
    parser.add_argument("-m", "--model", type=str, default="ferplus", help="Which model to run, either 'bertsquad' or 'ferplus'")

    args = parser.parse_args()

    if args.model == "ferplus":
        model = ferplus.Model
    elif args.model == "bertsquad":
        model = bertsquad.Model

    main(model, profile=args.profile, niter=args.niter, use_gpu = not args.cpu)

    # Cprofile
    # profFile = "fer10kGPU.prof"
    # cProfile.runctx("main(profile=False, niter=niter)", globals=globals(), locals=locals(), sort="cumulative", filename=profFile)
    # print("Profile at: ", profFile)

import cProfile
from contextlib import contextmanager
import ferplus
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


def runTest(model, state, niter=1):
    inputs = model.inputs()

    for repeat in range(niter):
        for testIn in inputs:
            processed = model.pre(testIn)

            # predict using the deployed model
            modelOut = model.run(state, processed)

            finalOut = model.post(modelOut)


def main(model, profile=False, niter=1, use_gpu=True):
    if profile:
        times = {}
        
        with timer("imports", times):
            model.imports()

        with timer("init", times):
            if use_gpu:
                state = model.init(profile=profile, provider="CUDAExecutionProvider")
            else:
                state = model.init(profile=profile, provider="CPUExecutionProvider")

        with timer("run", times):
            runTest(model, state, niter=niter)

        prof_file = state['session'].end_profiling()
        print("onnxruntime profile at: ", prof_file)
        print("Times: ")
        print(times)
    else:
        model.imports()
        if use_gpu:
            state = model.init(profile=profile, provider="CUDAExecutionProvider") 
        else:
            state = model.init(profile=profile, provider="CPUExecutionProvider") 
        runTest(model, state, niter=niter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for faas-like onnxruntime benchmarking")
    parser.add_argument("-c", "--cpu", action='store_true',
            help="use CPU execution provider (rather than the default CUDA provider)")
    parser.add_argument("-p", "--profile", action='store_true', help="Enable self-profiling")
    parser.add_argument("-n", "--niter", type=int, default=1, help="Number of test iterations to perform")

    args = parser.parse_args()

    model = ferplus.interface
    main(model, profile=args.profile, niter=args.niter, use_gpu = not args.cpu)

    # # Cprofile
    # # profFile = "fer10kGPU.prof"
    # # cProfile.runctx("main(profile=False, niter=niter)", globals=globals(), locals=locals(), sort="cumulative", filename=profFile)
    # # print("Profile at: ", profFile)

import cProfile
from contextlib import contextmanager
import ferplus
import time

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


def main(model, profile=False, niter=1):
    if profile:
        times = {}
        
        with timer("imports", times):
            model.imports()

        with timer("init", times):
            state = model.init(profile=profile)

        with timer("run", times):
            runTest(model, state, niter=niter)

        prof_file = state['session'].end_profiling()
        print("onnxruntime profile at: ", prof_file)
        print("Times: ")
        print(times)
    else:
        imports()
        state = initFer()
        runTest(state)


if __name__ == "__main__":
    model = ferplus.interface
    niter = 1

    # No profiling
    # main(niter = niter)

    # Internal Profiling
    main(model, profile=True, niter=niter)

    # Cprofile
    # profFile = "fer10kGPU.prof"
    # cProfile.runctx("main(profile=False, niter=niter)", globals=globals(), locals=locals(), sort="cumulative", filename=profFile)
    # print("Profile at: ", profFile)

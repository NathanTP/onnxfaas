# Helpful Docker Images
Building and running onnxruntime is a pain and has lots of tricky dependencies.
Your best bet is to work in the docker containers provided here. In particular,
you may as well use Dockerfile.build until image size becomes an issue (e.g.
deploying to a faas platform).

These dockerfiles and associated scripts are mostly taken from
onnxruntime/tools/ci\_build/github/linux/docker with some modifications:

1. We use miniconda for all our python stuff, it just simplifies dependencies
   and lets you goof around easier without having to rebuild the container when
   you inevitably mess something up.
2. We set the default user to your host user (instead of root) so that you can
   easily share files between the container/host without permissions issues.



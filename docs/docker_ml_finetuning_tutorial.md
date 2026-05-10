# Docker Tutorial for Machine Learning and Sketchformer Fine-Tuning
max_seq_len = 200
and the dataloader truncates sketches longer than that. So with current default fine-tuning, Sketchformer only sees up to 200 tokens/points, not 20,000 image pixels.

Rough attention size with num_heads=8:

sequence length N	N² pairs	attention memory per layer, batch=1, fp32
200	40,000	~1.3 MB
1,000	1,000,000	~32 MB
2,000	4,000,000	~128 MB
5,000	25,000,000	~800 MB
10,000	100,000,000	~3.2 GB
20,000	400,000,000	~12.8 GB
This guide is written for someone who has never used Docker before.


The goal is not only to explain Docker commands. The goal is to help you understand why Docker is useful for this project, how it relates to machine learning environments, and how you will use it during Sketchformer fine-tuning.

By the end, you should be able to answer:

- What is Docker?
- What is the difference between an image, a container, a Dockerfile, a volume, and a bind mount?
- Why do ML projects often use Docker?
- Why should this project use a separate Docker environment for Sketchformer training?
- How do you run fine-tuning while keeping datasets, checkpoints, and experiment outputs on your normal machine?

---

## 1. The Problem Docker Solves

Before Docker, a typical ML setup looked like this:

```text
Install Python
Install CUDA
Install cuDNN
Install TensorFlow or PyTorch
Install system packages
Install project packages
Run the training script
Something breaks
Try a different version
Something else breaks
```

That is especially painful in machine learning because ML projects often depend on very specific combinations:

```text
Python version
CUDA version
cuDNN version
TensorFlow or PyTorch version
NumPy version
Operating system libraries
GPU driver compatibility
```

Your project has exactly this kind of situation.

The current preprocessing pipeline is modern Python code. It uses tools like OpenCV, ControlNet auxiliary models, NumPy, scikit-learn, and dataset preparation scripts.

Sketchformer training, however, is older. The bundled Sketchformer dependency file asks for:

```text
tensorflow-gpu == 2.1
tensorflow_addons >= 0.6
numpy >= 1.16.4
```

The existing Sketchformer Dockerfile uses:

```text
nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
```

That means it was designed around an older TensorFlow GPU stack.

If you install that old TensorFlow stack into your current project environment, you risk breaking the preprocessing tools. Docker lets you keep the training stack isolated.

The practical split for this repo is:

```text
Current local environment:
  - preprocessing
  - image extraction
  - vectorization
  - stroke5 generation
  - Tok-Dict experiments

Docker training environment:
  - TensorFlow 2.1-era Sketchformer training
  - checkpoint restore
  - fine-tuning
  - GPU runtime
```

That is the main reason Docker matters here.

---

## 2. What Docker Is

Docker is a tool for packaging and running software in isolated environments called containers.

A container is like a lightweight, reproducible mini-computer that runs on your machine.

It contains:

- the operating system libraries your app needs
- Python
- Python packages
- system packages
- environment variables
- a command to run

It does not contain:

- a full separate physical machine
- your whole operating system
- your private files unless you explicitly mount them

A simple mental model:

```text
Your machine
  Docker
    Container
      Linux environment
      Python
      TensorFlow
      Your code
      Training command
```

Docker helps you say:

```text
"Run this project in this exact environment."
```

Instead of:

```text
"I hope my laptop has the right versions installed."
```

---

## 3. Docker vs Virtual Environments

You may already know Python virtual environments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A virtual environment isolates Python packages.

Docker isolates much more.

| Tool | Isolates Python packages | Isolates system packages | Helps with CUDA/cuDNN | Reproducible across machines |
|---|---:|---:|---:|---:|
| Python venv | yes | no | no | partly |
| Conda | yes | partly | partly | partly |
| Docker | yes | yes | yes, with correct GPU setup | yes |

For normal Python projects, a venv is often enough.

For ML training with old TensorFlow and GPU dependencies, Docker is usually much safer.

---

## 4. Core Docker Vocabulary

### Image

An image is a packaged environment.

Example:

```text
sketchformer-tf21:cuda10.1
```

An image is like a recipe that has already been cooked and frozen. You can start many containers from the same image.

### Container

A container is a running instance of an image.

If an image is like a class, a container is like an object.

```text
Image:
  sketchformer-tf21:cuda10.1

Containers:
  training-run-1
  training-run-2
  debug-shell
```

Containers are disposable. You should be comfortable deleting and recreating them.

### Dockerfile

A Dockerfile is the recipe used to build an image.

It says things like:

```text
Start from Ubuntu with CUDA
Install Python
Install pip packages
Set the working directory
```

This repo already has one here:

```text
sketchformer/dependencies/Dockerfile
```

### Build

Building means turning a Dockerfile into an image.

```bash
docker build -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

### Run

Running means starting a container from an image.

```bash
docker run --rm -it sketchformer-tf21:cuda10.1 bash
```

### Registry

A registry is a place where Docker images are stored.

Docker Hub is the common public registry.

Examples of images from registries:

```text
ubuntu:22.04
python:3.10-slim
nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
```

### Volume

A volume is persistent storage managed by Docker.

It is useful for caches and data that should survive after containers stop.

### Bind mount

A bind mount connects a normal folder on your machine to a folder inside the container.

For ML work, bind mounts are extremely important.

Example:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  ubuntu:22.04 bash
```

This means:

```text
On your machine:
  current project folder

Inside container:
  /workspace
```

Now the container can see your code and data.

---

## 5. The Most Important Docker Commands

Check Docker is installed:

```bash
docker --version
```

Run a tiny test container:

```bash
docker run hello-world
```

List images:

```bash
docker images
```

List running containers:

```bash
docker ps
```

List all containers, including stopped ones:

```bash
docker ps -a
```

Build an image:

```bash
docker build -t my-image-name:my-tag path/to/build/context
```

Run a shell inside an image:

```bash
docker run --rm -it my-image-name:my-tag bash
```

Stop a running container:

```bash
docker stop CONTAINER_ID_OR_NAME
```

Remove a stopped container:

```bash
docker rm CONTAINER_ID_OR_NAME
```

Remove an image:

```bash
docker rmi IMAGE_ID_OR_NAME
```

Show disk usage:

```bash
docker system df
```

Clean unused containers, networks, and dangling images:

```bash
docker system prune
```

Be careful with prune commands. They can remove things you forgot you still wanted.

---

## 6. Your First Docker Run Command

Try this:

```bash
docker run --rm -it ubuntu:22.04 bash
```

Breakdown:

```text
docker run
  Start a new container.

--rm
  Delete the container automatically when it exits.

-it
  Interactive terminal mode.

ubuntu:22.04
  Image name and tag.

bash
  Command to run inside the container.
```

Once inside the container:

```bash
pwd
ls
cat /etc/os-release
exit
```

Important idea:

```text
When you exit, the container is gone because you used --rm.
```

That is good. Containers should be easy to recreate.

---

## 7. The Filesystem Mental Model

The container has its own filesystem.

If you run:

```bash
docker run --rm -it ubuntu:22.04 bash
```

and create a file inside it:

```bash
echo "hello" > inside_container.txt
```

then exit, that file disappears.

This surprises many beginners.

Containers are temporary unless you persist data using:

- bind mounts
- Docker volumes
- copying files out
- writing to a network service

For ML, you almost always use bind mounts for:

- source code
- datasets
- checkpoints
- logs
- generated artifacts

You do not want training checkpoints trapped inside a deleted container.

---

## 8. Bind Mounts for ML Projects

Bind mounts are how your normal project files appear inside the container.

Example:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  ubuntu:22.04 bash
```

Inside the container:

```bash
cd /workspace
ls
```

You should see your project files.

The path mapping is:

```text
Host machine path:
  /home/.../text-to-sketch

Container path:
  /workspace
```

This is one of the most important Docker concepts for ML:

```text
The container supplies the environment.
Your host machine supplies the data, code, and persistent outputs.
```

That is exactly what you want.

### Read-only bind mount

Sometimes you want the container to read a folder but not modify it:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace,readonly \
  ubuntu:22.04 bash
```

For fine-tuning, your code and output directories usually need to be writable.

For credentials, read-only is safer.

Example:

```bash
--mount type=bind,source="$HOME/.kaggle",target=/root/.kaggle,readonly
```

---

## 9. Docker Volumes for ML Caches

A Docker volume is storage managed by Docker.

Create a volume:

```bash
docker volume create hf-cache
```

Use it in a container:

```bash
docker run --rm -it \
  --mount type=volume,source=hf-cache,target=/root/.cache/huggingface \
  python:3.10 bash
```

This is useful for:

- Hugging Face model caches
- pip caches
- dataset caches
- experiment tracking service caches

For this project, bind mounts are more important than volumes because you want clear local folders:

```text
data/
experiments/
pretrained/
```

But volumes are useful once you start downloading large pretrained assets repeatedly.

---

## 10. Dockerfile Basics

A Dockerfile describes how to build an image.

Simple example:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

Line by line:

```text
FROM python:3.10-slim
  Start from an existing Python image.

WORKDIR /app
  Set the default working directory inside the image.

COPY requirements.txt .
  Copy dependency file into the image.

RUN pip install --no-cache-dir -r requirements.txt
  Install Python packages while building the image.

COPY . .
  Copy project source into the image.

CMD ["python", "train.py"]
  Default command when a container starts.
```

For ML, you often do not copy data into the image.

Bad idea:

```dockerfile
COPY data/ /app/data/
```

Why bad?

- images become huge
- rebuilding becomes slow
- datasets change more often than dependencies
- you risk accidentally packaging private data

Better idea:

```text
Put code and dependencies in the image.
Mount data at runtime.
```

---

## 11. The Existing Sketchformer Dockerfile

This repo already includes:

```text
sketchformer/dependencies/Dockerfile
```

It starts with:

```dockerfile
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
```

That means:

```text
Use Ubuntu 18.04
Use CUDA 10.1 runtime
Use cuDNN 7 runtime
Expect NVIDIA GPU support
```

Then it installs system packages:

```dockerfile
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    build-essential \
    libblas-dev \
    libhdf5-serial-dev \
    python3-dev \
    python3-pip \
    git
```

These are Linux packages needed to compile or run Python ML dependencies.

Then it installs Python packages:

```dockerfile
ADD ./requirements.txt ./
ADD ./git-requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install -r git-requirements.txt
```

Important detail:

```text
This Dockerfile installs the Sketchformer dependencies.
It does not copy the whole text-to-sketch repository into the image.
```

That is okay. For development and fine-tuning, you can mount the repo at runtime.

---

## 12. Images and Build Cache

When you build an image, Docker executes the Dockerfile one step at a time.

Each step becomes a cached layer.

Example:

```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

This is common because dependencies change less often than source code.

If only your Python files change, Docker can reuse the dependency installation layer.

In ML projects, build cache matters because installing TensorFlow, PyTorch, CUDA-related packages, and scientific packages can be slow.

---

## 13. Docker and GPUs

Docker does not replace your GPU driver.

For GPU training, you still need:

```text
Host machine:
  NVIDIA GPU
  NVIDIA driver
  Docker
  NVIDIA Container Toolkit

Container:
  CUDA runtime libraries
  TensorFlow or PyTorch
```

The container uses the host GPU through Docker.

To run with GPU access:

```bash
docker run --rm -it --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

For the existing Sketchformer image, the command shape is:

```bash
docker run --rm -it --gpus all sketchformer-tf21:cuda10.1 bash
```

Inside the container, test TensorFlow:

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.experimental.list_physical_devices("GPU"))
PY
```

If this shows a GPU, TensorFlow can see the GPU.

If it does not, common causes are:

- NVIDIA driver not installed on host
- NVIDIA Container Toolkit not installed
- Docker daemon not configured for NVIDIA runtime
- incompatible CUDA/TensorFlow/driver combination
- running on a machine without an NVIDIA GPU

For this repo, the existing Sketchformer Dockerfile uses CUDA 10.1. That is old. It may work if your host NVIDIA driver supports running CUDA 10.1 containers, but if it fails, you may need to modernize the Dockerfile or use a CPU/non-Docker training environment.

---

## 14. Docker and Environment Variables

Environment variables let you pass configuration into a container.

Example:

```bash
docker run --rm -it \
  -e KAGGLE_USERNAME="your_username" \
  -e KAGGLE_KEY="your_key" \
  python:3.10 bash
```

If you have a `.env` file:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
KAGGLE_DATASET=diraizel/anime-images-dataset
```

You can pass it:

```bash
docker run --rm -it --env-file .env python:3.10 bash
```

For secrets:

- do not hard-code them in a Dockerfile
- do not commit `.env`
- prefer mounting credential files read-only
- keep secrets on the host machine when possible

For Kaggle credentials, a safe pattern is:

```bash
--mount type=bind,source="$HOME/.kaggle",target=/root/.kaggle,readonly
```

---

## 15. Docker and Ports

Ports matter when the container runs a server.

Examples:

- Jupyter Notebook
- TensorBoard
- FastAPI
- Streamlit
- a web app

If TensorBoard runs inside the container on port `6006`, expose it like this:

```bash
docker run --rm -it \
  -p 6006:6006 \
  my-ml-image bash
```

This maps:

```text
Host machine:
  http://localhost:6006

Container:
  port 6006
```

For plain training scripts, you often do not need ports.

For experiment visualization, TensorBoard may need a port.

---

## 16. Docker and ML Reproducibility

Docker helps ML reproducibility because it fixes the environment.

But Docker alone does not make an experiment fully reproducible.

For a reproducible fine-tuning run, record:

- Docker image name and tag
- Git commit
- dataset version
- preprocessing settings
- training command
- hparams
- random seed
- checkpoint path
- output directory
- GPU type
- major library versions

For this repo, a useful experiment record might include:

```text
Image:
  sketchformer-tf21:cuda10.1

Code:
  git commit <commit>

Dataset:
  data/sketchformer-ready-data/stroke3

Pretrained checkpoint:
  pretrained/sketchformer/weights/ckpt-100

Output:
  experiments/sketch-transformer-tf2-anime-finetune-docker

Command:
  python train.py sketch-transformer-tf2 ...
```

Docker fixes the environment. You still need to manage data and experiment metadata carefully.

---

## 17. How Docker Fits This Project

This repo has two major workflows.

### Workflow A: Preprocessing

Input:

```text
raw anime images
```

Pipeline:

```text
lineart extraction
vectorization
stroke ordering
kinematics
stroke5 formatting
Tok-Dict encoding
Sketchformer-ready dataset preparation
```

Output:

```text
data/processed/
data/sketchformer-ready-data/
```

This can stay in your current local Python environment.

### Workflow B: Fine-tuning

Input:

```text
Sketchformer-ready dataset
pretrained checkpoint
training hparams
```

Pipeline:

```text
start TensorFlow Sketchformer environment
restore checkpoint with --resume
train on anime dataset
write new checkpoints
```

Output:

```text
experiments/sketch-transformer-tf2-<experiment-id>/weights/
```

This is where Docker helps most.

### The project architecture

```text
text-to-sketch/
  data/
    raw/
    processed/
    sketchformer-ready-data/

  sketchformer/
    train.py
    dependencies/
      Dockerfile
      requirements.txt

  experiments/
    sketch-transformer-tf2-anime-finetune/
      weights/

  pretrained/
    sketchformer/
      weights/
```

Recommended environment split:

```text
Local .venv:
  run preprocessing scripts

Docker container:
  run sketchformer/train.py
```

---

## 18. Why Not Put Everything in One Environment?

Because the preprocessing and training stacks want different dependency worlds.

Preprocessing stack:

```text
modern Python
OpenCV
ControlNet auxiliary packages
scikit-learn
Torch-related packages
newer NumPy
```

Sketchformer training stack:

```text
old TensorFlow GPU
old CUDA/cuDNN expectation
older package assumptions
```

If you combine them, you may get errors like:

```text
No matching distribution found for tensorflow-gpu==2.1
numpy version conflict
CUDA library not found
ImportError from TensorFlow internals
```

Docker avoids this by saying:

```text
The host environment can be modern.
The training container can be old and stable.
```

That is the clean approach.

---

## 19. Build the Sketchformer Docker Image

From the project root:

```bash
docker build -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

Breakdown:

```text
docker build
  Build an image.

-t sketchformer-tf21:cuda10.1
  Tag the image with a readable name.

sketchformer/dependencies
  Use this folder as the build context.
  Docker will find Dockerfile inside it.
```

After build:

```bash
docker images
```

You should see:

```text
sketchformer-tf21   cuda10.1
```

If build fails, read the first real error above the final failure line. Docker output can be noisy, but the meaningful error is usually earlier.

---

## 20. Start a Training Shell for This Repo

From the project root:

```bash
mkdir -p experiments pretrained
```

Then start a container:

```bash
docker run --rm -it \
  --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

Breakdown:

```text
--rm
  Remove the container when it exits.

-it
  Interactive terminal.

--gpus all
  Give the container access to all visible NVIDIA GPUs.

--mount type=bind,source="$(pwd)",target=/workspace
  Mount this repo into /workspace inside the container.

--env-file .env
  Pass environment variables from .env into the container.

-w /workspace/sketchformer
  Start inside the Sketchformer source directory.

sketchformer-tf21:cuda10.1
  Use the image you built.

bash
  Open a shell.
```

Inside the container:

```bash
pwd
ls
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

You should be in:

```text
/workspace/sketchformer
```

The repo root is:

```text
/workspace
```

---

## 21. Path Mapping for This Project

When you mount the repo with:

```bash
--mount type=bind,source="$(pwd)",target=/workspace
```

paths change like this:

| Host path | Container path |
|---|---|
| `./README.md` | `/workspace/README.md` |
| `./data` | `/workspace/data` |
| `./sketchformer/train.py` | `/workspace/sketchformer/train.py` |
| `./experiments` | `/workspace/experiments` |
| `./pretrained` | `/workspace/pretrained` |

This matters because training commands inside Docker should use container paths.

Host command path:

```text
data/sketchformer-ready-data/stroke3
```

Container command path:

```text
/workspace/data/sketchformer-ready-data/stroke3
```

---

## 22. Fine-Tuning Flow With Docker

The full flow should look like this:

```text
1. Use local preprocessing environment
   raw anime images -> stroke/sketchformer-ready dataset

2. Put pretrained weights somewhere persistent
   pretrained/sketchformer/weights/ckpt-100

3. Build Docker image
   sketchformer-tf21:cuda10.1

4. Start Docker container with repo mounted
   /workspace points to this repo

5. Run Sketchformer training inside container
   python train.py ...

6. Save new fine-tuned checkpoints
   /workspace/experiments/...

7. Exit container
   outputs remain on host machine
```

The important idea:

```text
The container can disappear.
The experiment output remains in ./experiments on your host.
```

---

## 23. Example Fine-Tuning Command

Inside the Docker container:

```bash
cd /workspace/sketchformer
```

Then run:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-docker \
  --resume /workspace/pretrained/sketchformer/weights/ckpt-100 \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

What each part means:

```text
python train.py sketch-transformer-tf2
  Run the Sketchformer training script with the transformer model.

--dataset /workspace/data/sketchformer-ready-data/stroke3
  Use the prepared anime sketch dataset.

--output-dir /workspace/experiments
  Save experiment outputs to the mounted host folder.

--id anime-finetune-docker
  Name this run.

--resume /workspace/pretrained/sketchformer/weights/ckpt-100
  Load pretrained checkpoint weights before continuing training.

--hparams "do_classification=False"
  Disable classification for unlabeled anime sketches.

--data-hparams "use_continuous_data=True,max_seq_len=500"
  Use continuous stroke data and longer sequences.

--base-hparams "batch_size=8,num_epochs=10"
  Set training batch size and number of epochs.
```

Expected output folder:

```text
experiments/sketch-transformer-tf2-anime-finetune-docker/
  config.json
  plots/
  tmp/
  weights/
```

That output lives on your host machine because `/workspace/experiments` is bind-mounted from `./experiments`.

---

## 24. Fine-Tuning From Scratch vs From Pretrained

Training from scratch:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-train-from-scratch \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

Fine-tuning from pretrained:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-from-pretrained \
  --resume /workspace/pretrained/sketchformer/weights/ckpt-100 \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

The difference is:

```text
without --resume:
  start from random weights

with --resume:
  load pretrained weights and continue training
```

For your objective, fine-tuning usually means using `--resume`.

---

## 25. Preserving Original Pretrained Weights

Do not save new fine-tuning outputs into the pretrained checkpoint folder.

Use a separate output directory:

```text
pretrained/
  sketchformer/
    weights/
      ckpt-100.index
      ckpt-100.data-00000-of-00001

experiments/
  sketch-transformer-tf2-anime-finetune-docker/
    weights/
      new checkpoints
```

Good:

```bash
--resume /workspace/pretrained/sketchformer/weights/ckpt-100
--output-dir /workspace/experiments
--id anime-finetune-docker
```

Bad:

```bash
--resume /workspace/pretrained/sketchformer/weights/ckpt-100
--output-dir /workspace/pretrained/sketchformer
```

The good version reads pretrained weights from `pretrained/` and writes new weights to `experiments/`.

---

## 26. Inspect Before Training

GPU training is expensive. Before fine-tuning, always inspect:

- dataset path exists
- `meta.npz` exists
- `train_*.npz` exists
- `valid.npz` exists
- `test.npz` exists
- sequence lengths are reasonable
- labels are handled correctly
- batch size fits GPU memory

Example checks inside the container:

```bash
ls /workspace/data/sketchformer-ready-data/stroke3
ls /workspace/pretrained/sketchformer/weights
```

If you have an inspection script, run it before training:

```bash
python /workspace/scripts/inspect_sketchformer_data.py \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --max-seq-len 500
```

This prevents wasting time on a training run that fails after loading the first batch.

---

## 27. Running Without GPU

If you do not have a working NVIDIA Docker setup, you can still start the container without GPU:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

This removes:

```text
--gpus all
```

CPU training will be much slower. It can still be useful for:

- checking imports
- checking command syntax
- inspecting data loading
- running very tiny smoke tests

For real fine-tuning, GPU is strongly preferred.

---

## 28. Avoiding Root-Owned Output Files

On Linux, Docker containers often run as root by default.

If the container writes files into your mounted `experiments/` folder, those files may become owned by root on your host machine.

You can avoid this by running the container as your current user:

```bash
docker run --rm -it \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

If this causes permission issues inside the container, use the simpler command first and fix ownership afterward:

```bash
sudo chown -R "$USER:$USER" experiments
```

For beginners, start simple. Once you understand the flow, improve permissions.

---

## 29. Docker Compose Optional Shortcut

Docker Compose lets you save long `docker run` commands in a YAML file.

Example `compose.yaml`:

```yaml
services:
  sketchformer:
    build:
      context: ./sketchformer/dependencies
    image: sketchformer-tf21:cuda10.1
    working_dir: /workspace/sketchformer
    volumes:
      - .:/workspace
    env_file:
      - .env
    stdin_open: true
    tty: true
```

Then start a shell:

```bash
docker compose run --rm sketchformer bash
```

GPU support in Compose depends on your Docker and NVIDIA runtime setup. For beginners, the explicit `docker run --gpus all ...` command is easier to debug.

---

## 30. What Goes Into the Image vs What Gets Mounted

Put these in the image:

- OS libraries
- Python version
- TensorFlow/PyTorch
- Python dependencies
- command-line tools needed by training

Mount these at runtime:

- source code during development
- datasets
- checkpoints
- experiment outputs
- credentials
- logs

For this project:

| Thing | Image or mount? | Why |
|---|---|---|
| TensorFlow 2.1 | image | dependency environment |
| CUDA runtime | image | GPU library compatibility |
| `sketchformer/train.py` | mount | code changes during development |
| `data/` | mount | too large and changes often |
| `experiments/` | mount | must persist after container exits |
| pretrained checkpoints | mount | large and should not be baked into image |
| `.env` | env file | secrets/config should not be baked into image |

---

## 31. Common Docker Mistakes in ML

### Mistake 1: Saving checkpoints only inside the container

Bad:

```text
output-dir=/tmp/experiments
```

If `/tmp/experiments` is inside the container and not mounted, you may lose it.

Good:

```text
output-dir=/workspace/experiments
```

where `/workspace` is a bind mount.

### Mistake 2: Building data into the image

Bad:

```dockerfile
COPY data/ /app/data/
```

Good:

```bash
--mount type=bind,source="$(pwd)/data",target=/workspace/data
```

### Mistake 3: Confusing host paths and container paths

Bad inside container:

```bash
python train.py --dataset /home/naolselemon/Documents/.../data
```

Good inside container:

```bash
python train.py --dataset /workspace/data/sketchformer-ready-data/stroke3
```

### Mistake 4: Assuming Docker includes your GPU automatically

You must use:

```text
--gpus all
```

and your host must have NVIDIA Container Toolkit configured.

### Mistake 5: Mixing preprocessing and training dependencies

Do not install old TensorFlow GPU dependencies into the current preprocessing `.venv`.

Use Docker or a separate environment for Sketchformer training.

---

## 32. Debugging Checklist

### Docker command fails with "Cannot connect to the Docker daemon"

Check Docker is running:

```bash
docker ps
```

On Linux, you may need:

```bash
sudo systemctl status docker
```

### Permission denied using Docker

You may need to run with `sudo` or add your user to the Docker group.

Be careful: membership in the Docker group is powerful. Treat it like admin access.

### Build fails while installing packages

Try:

```bash
docker build --no-cache -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

Also check:

- internet connection
- package versions
- old Python package compatibility
- whether the base image still supports the required package sources

### `--gpus all` fails

Check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If host `nvidia-smi` works but Docker GPU does not, the issue is probably NVIDIA Container Toolkit or Docker runtime configuration.

### TensorFlow cannot see GPU

Inside the container:

```bash
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.experimental.list_physical_devices("GPU"))
PY
```

If the GPU list is empty:

- verify `--gpus all`
- verify NVIDIA Container Toolkit
- verify CUDA/TensorFlow compatibility
- test with `nvidia-smi` inside the container

### Training runs out of memory

Lower batch size:

```bash
--base-hparams "batch_size=4,num_epochs=10"
```

You can also reduce sequence length:

```bash
--data-hparams "use_continuous_data=True,max_seq_len=200"
```

### Checkpoint restore fails

Likely causes:

- wrong checkpoint path
- missing `.index` or `.data` file
- changed model architecture
- changed continuous/discrete data mode
- changed sequence assumptions
- checkpoint from a different model

Keep architecture-related hparams compatible when using `--resume`.

---

## 33. Practical Learning Exercises

Do these in order.

### Exercise 1: Run a temporary Ubuntu container

```bash
docker run --rm -it ubuntu:22.04 bash
```

Inside:

```bash
cat /etc/os-release
exit
```

### Exercise 2: Mount this repo

From the project root:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  ubuntu:22.04 bash
```

Inside:

```bash
ls /workspace
exit
```

### Exercise 3: Build the Sketchformer image

```bash
docker build -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

### Exercise 4: Open a shell in the image

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

Inside:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
exit
```

### Exercise 5: Try GPU visibility

```bash
docker run --rm -it \
  --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

Inside:

```bash
nvidia-smi
python - <<'PY'
import tensorflow as tf
print(tf.config.experimental.list_physical_devices("GPU"))
PY
exit
```

### Exercise 6: Run a tiny training dry run

Use this only after your dataset and checkpoint paths exist.

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id docker-smoke-test \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=200" \
  --base-hparams "batch_size=2,num_epochs=1"
```

The purpose is not model quality. The purpose is to prove that:

- imports work
- data loading works
- output writing works
- Docker path mapping works

---

## 34. Recommended Fine-Tuning Routine

Use this routine when you are ready for real training.

### Step 1: Prepare data locally

Use the current local environment for preprocessing:

```bash
source .venv/bin/activate
python scripts/download_data.py
python scripts/extract_sketches.py
python scripts/run_pipeline.py
python scripts/prepare_anime_data.py
```

The exact commands may change as your pipeline matures, but the idea stays:

```text
Prepare Sketchformer-ready .npz files before training.
```

### Step 2: Place pretrained checkpoint

Use a persistent host folder:

```text
pretrained/sketchformer/weights/
```

Example:

```text
pretrained/sketchformer/weights/ckpt-100.index
pretrained/sketchformer/weights/ckpt-100.data-00000-of-00001
```

### Step 3: Build or rebuild image

```bash
docker build -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

### Step 4: Start training shell

```bash
docker run --rm -it \
  --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

### Step 5: Verify inside container

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
ls /workspace/data/sketchformer-ready-data/stroke3
ls /workspace/pretrained/sketchformer/weights
```

### Step 6: Run fine-tuning

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-docker \
  --resume /workspace/pretrained/sketchformer/weights/ckpt-100 \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

### Step 7: Check outputs

On your host machine:

```bash
ls experiments/sketch-transformer-tf2-anime-finetune-docker
ls experiments/sketch-transformer-tf2-anime-finetune-docker/weights
```

### Step 8: Resume if interrupted

Inside Docker:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-docker \
  --resume latest \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

`--resume latest` loads the latest checkpoint from the current experiment's output folder.

---

## 35. How to Think About Docker During Fine-Tuning

Think of the Docker image as:

```text
The lab equipment
```

Think of the container as:

```text
A temporary lab session
```

Think of mounted folders as:

```text
The notebooks, datasets, and saved results you take out of the lab
```

The lab session can end. Your results should remain.

In practical terms:

```text
Image:
  sketchformer-tf21:cuda10.1

Container:
  one training session

Mounted input:
  data/
  pretrained/

Mounted output:
  experiments/
```

---

## 36. What Docker Does Not Solve

Docker is powerful, but it does not solve everything.

It does not:

- magically fix incompatible model checkpoints
- make a weak GPU faster
- choose correct hyperparameters
- prevent out-of-memory errors
- guarantee deterministic training
- replace dataset validation
- replace experiment tracking
- remove the need to understand paths

Docker solves the environment problem.

You still need good ML practice.

---

## 37. A Simple Rule for This Project

Use this rule:

```text
If the task is data preprocessing, use the local project environment.
If the task is Sketchformer TensorFlow training, use Docker.
```

Examples:

| Task | Recommended environment |
|---|---|
| Download anime images | local `.venv` |
| Extract lineart sketches | local `.venv` |
| Convert sketches to stroke5 | local `.venv` |
| Build Tok-Dict codebook | local `.venv` |
| Prepare Sketchformer `.npz` dataset | local `.venv` |
| Restore TensorFlow checkpoint | Docker |
| Fine-tune Sketchformer | Docker |
| Resume fine-tuning | Docker |
| Save new model weights | Docker writing to mounted `experiments/` |

---

## 38. Minimal Command Cheat Sheet for This Repo

Build image:

```bash
docker build -t sketchformer-tf21:cuda10.1 sketchformer/dependencies
```

Run shell with GPU:

```bash
docker run --rm -it \
  --gpus all \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

Run shell without GPU:

```bash
docker run --rm -it \
  --mount type=bind,source="$(pwd)",target=/workspace \
  --env-file .env \
  -w /workspace/sketchformer \
  sketchformer-tf21:cuda10.1 \
  bash
```

Check TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Check GPU:

```bash
nvidia-smi
```

Fine-tune:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-docker \
  --resume /workspace/pretrained/sketchformer/weights/ckpt-100 \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

Resume current experiment:

```bash
python train.py sketch-transformer-tf2 \
  --dataset /workspace/data/sketchformer-ready-data/stroke3 \
  --output-dir /workspace/experiments \
  --id anime-finetune-docker \
  --resume latest \
  --hparams "do_classification=False" \
  --data-hparams "use_continuous_data=True,max_seq_len=500" \
  --base-hparams "batch_size=8,num_epochs=10"
```

---

## 39. Recommended Next Files to Add Later

This tutorial explains Docker. Later, the project may benefit from adding:

```text
Dockerfile.sketchformer
compose.yaml
.dockerignore
scripts/train_sketchformer_continuous_finetune.py
scripts/inspect_sketchformer_data.py
docs/fine_tuning_runbook.md
```

Suggested purpose:

| File | Purpose |
|---|---|
| `Dockerfile.sketchformer` | modernized or project-specific training image |
| `compose.yaml` | shorter repeatable Docker run commands |
| `.dockerignore` | faster image builds, avoid copying data/checkpoints |
| `scripts/inspect_sketchformer_data.py` | validate data before GPU runs |
| `scripts/train_sketchformer_continuous_finetune.py` | safer training launcher |
| `docs/fine_tuning_runbook.md` | exact experiment commands |

Do not start by overbuilding Docker tooling. First learn the image, container, mount, GPU, and checkpoint flow.

---

## 40. References

Official references worth keeping nearby:

- Docker Engine installation: https://docs.docker.com/engine/install/
- Docker Engine on Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- Docker bind mounts: https://docs.docker.com/engine/storage/bind-mounts/
- Docker volumes: https://docs.docker.com/engine/storage/volumes/
- NVIDIA Container Toolkit installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Project-local references:

- `README.md`
- `sketchformer/dependencies/Dockerfile`
- `sketchformer/dependencies/requirements.txt`
- `study/fine-tune-strategy/fine_tuning_pretrained_weights.md`
- `study/continous-mode-fine-tuning-strategies/implementation_checklist_and_environment_plan.md`

---

## Final Mental Model

Use Docker to freeze the training environment.

Use bind mounts to connect that environment to your real project files.

Use the local `.venv` for preprocessing.

Use the Docker Sketchformer environment for TensorFlow fine-tuning.

Keep datasets, pretrained checkpoints, and experiment outputs outside the container so they survive after the container exits.

That is the whole pattern:

```text
Local machine:
  code
  data
  pretrained weights
  experiment outputs

Docker image:
  operating system
  CUDA runtime
  Python
  TensorFlow
  Sketchformer dependencies

Docker container:
  temporary fine-tuning session
```


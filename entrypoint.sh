set -e

# ldconfig

echo 'PIP freeze (subset):'
pip freeze | grep nvidia
pip freeze | grep jax

echo GCP instance:
echo "Name:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name || echo NA)"
echo "Hostname: $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/hostname || echo NA)"
echo "ID:       $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id || echo NA)"
echo "Zone:     $(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone || echo NA)"
echo

echo GPUs:
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

# # Start Ollama server in the background
# ollama serve &

# # Wait a moment to ensure Ollama is up
# sleep 5

# # Pull the model (optional)
# ollama pull llava:7b
# ollama run llava:7b

xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@"
# xvfb-run "$@"
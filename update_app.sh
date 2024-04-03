# Fetch the code from GitHub
git pull origin main

# Install all dependencies
pip install -r requirements.txt

# Rebuild docker containers
bash scripts/docker/build.sh

# Delete old images
docker system prune -f

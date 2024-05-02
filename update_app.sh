# Fetch the code from GitHub
git pull origin main

# Install all dependencies
pip install -r requirements.txt

# Rebuild docker containers
docker pull tnodecode/spine-nginx
docker pull tnodecode/spineui


# Delete old images
docker system prune -f

import sys
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id=sys.argv[1], allow_patterns=[sys.argv[2]])
print(path)

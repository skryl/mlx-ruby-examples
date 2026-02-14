#!/usr/bin/env python3

import sys
from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
filename = sys.argv[2]
path = hf_hub_download(repo_id=repo_id, filename=filename)
print(path)

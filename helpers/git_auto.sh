#!/bin/bash

# add changes
git add .

# commit with message (default if none provided)
commit_message=${1:-"generic update"}
git commit -m "$commit_message"

# push to remote
git push 
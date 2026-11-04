#!/bin/bash
# Downloads bAbI v1.2 (Weston et al., 2015; CC BY 3.0) into data/public/babi/tasks_1-20_v1-2.
# The original Facebook link is gone; this is the Keras mirror. Only qa1-3 are used.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/public/babi
A=data/public/babi/babi_tasks_1-20_v1-2.tar.gz
[ -f $A ] || curl -L -o $A https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
echo "84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c  $A" | sha256sum -c -
tar -xzf $A -C data/public/babi

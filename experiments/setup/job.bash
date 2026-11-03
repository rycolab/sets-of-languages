#!/bin/bash
set -euo pipefail
module load eth_proxy
echo '==== Start of job ===='
exec /usr/bin/time -v bash experiments/setup.bash

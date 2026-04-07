#!/usr/bin/env bash
set -euo pipefail
cd /ceph/build.u2204
../src/vstart.sh --debug --new -x --localhost --short --bluestore
sleep 5
bin/ceph -s
bin/ceph osd pool create test_pool 8
bin/rados -p test_pool put test_obj /etc/hostname
bin/rados -p test_pool get test_obj /tmp/out
diff /etc/hostname /tmp/out
echo VSTART_OK
../src/stop.sh

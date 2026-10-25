#!/usr/bin/env bash
# Run on the same Linux host that will execute the compiled test binaries.
set -euo pipefail
required=${1:?Usage: check-cpu-features.sh 'avx2 fma f16c ...'}
cpuinfo=${CPUINFO_PATH:-/proc/cpuinfo}
# Require every advertised processor to support every enabled feature.
awk -v required="$required" '
  BEGIN { n = split(required, features, " "); supported = 1 }
  /^flags[[:space:]]*:/ {
    processors++
    flags = " " $0 " "
    for (i = 1; i <= n; i++) {
      if (index(flags, " " features[i] " ") == 0) {
        print "Missing CPU feature: " features[i] > "/dev/stderr"
        supported = 0
      }
    }
  }
  END { if (!processors || !supported) exit 1 }
' "$cpuinfo"

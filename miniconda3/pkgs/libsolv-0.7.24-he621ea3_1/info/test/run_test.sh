

set -ex



test -f ${PREFIX}/lib/libsolv${SHLIB_EXT}
test -f ${PREFIX}/lib/libsolvext${SHLIB_EXT}
test -f ${PREFIX}/lib/libsolv.so.1
test -f ${PREFIX}/include/solv/repo.h
dumpsolv -h
exit 0

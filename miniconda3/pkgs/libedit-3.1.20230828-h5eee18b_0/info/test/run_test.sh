

set -ex



test -f $PREFIX/lib/pkgconfig/libedit.pc
test -f $PREFIX/lib/libedit.so
exit 0

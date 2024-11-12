

set -ex



test -f $PREFIX/lib/libicudata.a
test -f $PREFIX/lib/libicudata.so.73.1
test -f $PREFIX/lib/libicui18n.a
test -f $PREFIX/lib/libicui18n.so.73.1
test -f $PREFIX/lib/libicuio.a
test -f $PREFIX/lib/libicuio.so.73.1
test -f $PREFIX/lib/libicutest.a
test -f $PREFIX/lib/libicutest.so.73.1
test -f $PREFIX/lib/libicutu.a
test -f $PREFIX/lib/libicutu.so.73.1
test -f $PREFIX/lib/libicuuc.a
test -f $PREFIX/lib/libicuuc.so.73.1
genbrk --help
gencfu --help
gencnval --help
gendict --help
icuinfo --help
icu-config --help
makeconv gb-18030-2000.ucm
genrb de.txt
echo "de.res" > list.txt
pkgdata -p mybundle list.txt
exit 0

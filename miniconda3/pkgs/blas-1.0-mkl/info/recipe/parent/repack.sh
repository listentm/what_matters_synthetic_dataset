#!/bin/bash
set -ex

cp $SRC_DIR/$PKG_NAME/info/LICENSE.txt $SRC_DIR
# for subpackages, we have named our extracted locations according to the subpackage name
#    That's what this $PKG_NAME is doing - picking the right subfolder to rsync

if [[ `uname` == "Linux" || `uname` == "Darwin" ]]; then
    cp -rv $SRC_DIR/$PKG_NAME/* "$PREFIX/"
else
    src=$(cygpath -u "$SRC_DIR/$PKG_NAME/")
    cp -rv $src/* "$PREFIX/"
fi

rm -rf $PREFIX/info

# Not necessary anymore.  Intel has been stripping these out themselves since 2018.0.1
#
# if [[ $(uname) == Darwin && -d $SRC_DIR/$PKG_NAME/lib ]]; then
#   # Strip off support for PPC - saves about 100 MB
#   if [ -d "$PREFIX/lib/intel64" ]; then
#       rm -rf "$SRC_DIR/$PKG_NAME/lib/intel64"
#       rm -rf "$PREFIX/lib/intel64"
#   fi
#   python $RECIPE_DIR/deuniversalize.py --ignore-wrong-arch $SRC_DIR/$PKG_NAME/lib/* $PREFIX/lib

# fi

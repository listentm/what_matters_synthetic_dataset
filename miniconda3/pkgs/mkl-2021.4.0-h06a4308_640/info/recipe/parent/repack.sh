#!/bin/bash
set -ex

# for subpackages, we have named our extracted locations according to the subpackage name
#    That's what this $PKG_NAME is doing - picking the right subfolder to rsync

src="$SRC_DIR/$PKG_NAME"

# # The license files are no longer manually packed but are included via the meta.yaml
# # not all packages have the license file.  Copy it from mkl, where we know it exists
# cp -f "$SRC_DIR/mkl/info/licenses/license.txt" "$SRC_DIR"
# # ro by default.  Makes installations not cleanly removable.
# chmod 664 "$SRC_DIR/license.txt"

cp -rv "$src"/* "$PREFIX/"

# replace old info folder with our new regenerated one
rm -rf "$PREFIX/info"

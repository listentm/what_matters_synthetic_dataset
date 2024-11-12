# not all packages have the license file.  Copy it from mkl, where we know it exists
cp -f $SRC_DIR/mkl/info/licenses/license.txt $SRC_DIR
# ro by default.  Makes installations not cleanly removable.
chmod 664 $SRC_DIR/license.txt

cp $RECIPE_DIR/site.cfg $PREFIX/site.cfg
echo library_dirs = $PREFIX/lib  >> $PREFIX/site.cfg
echo include_dirs = $PREFIX/include  >> $PREFIX/site.cfg

cp -rv $SRC_DIR/$PKG_NAME/ $PREFIX
rm -rf $PREFIX/info

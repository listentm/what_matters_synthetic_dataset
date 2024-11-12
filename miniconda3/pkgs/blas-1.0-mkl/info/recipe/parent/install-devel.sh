if [[ `uname` == "Linux" || `uname` == "Darwin" ]]; then
    cp $RECIPE_DIR/site.cfg $PREFIX/site.cfg
    echo library_dirs = $PREFIX/lib  >> $PREFIX/site.cfg
    echo include_dirs = $PREFIX/include  >> $PREFIX/site.cfg
else
    cp $RECIPE_DIR/site.cfg $PREFIX/site.cfg
    echo library_dirs = $(cygpath -w $LIBRARY_LIB)  >> $PREFIX/site.cfg
    echo include_dirs = $(cygpath -w $LIBRARY_INC) >> $PREFIX/site.cfg
fi

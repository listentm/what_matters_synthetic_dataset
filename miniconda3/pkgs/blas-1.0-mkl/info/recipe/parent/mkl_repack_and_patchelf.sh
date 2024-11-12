bash $RECIPE_DIR/repack.sh

# this was an attempt to fix:
#   ERROR (mkl,lib/libmkl_ao_worker.so): runpaths ['$ORIGIN/'] found in /opt/conda/conda-bld/intel_repacks_1525992717358/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold/lib/libmkl_ao_worker.so
# and
#   ERROR (mkl,lib/libmkl_ao_worker.so): did not find - or even know where to look for: libiomp5.so

# there is one outstanding unresolved lib that we've asked Intel about:

#   ERROR (mkl,lib/libmkl_ao_worker.so): did not find - or even know where to look for: libcoi_device.so.0

# if [[ $target_platform == linux-64 ]]; then
#     LIBRARY=$PREFIX/lib/libmkl_ao_worker.so
#     OLD_RPATH=$(patchelf --print-rpath $LIBRARY)
#     patchelf --remove-rpath $LIBRARY
#     patchelf --set-rpath '$ORIGIN':$OLD_RPATH $LIBRARY
# fi

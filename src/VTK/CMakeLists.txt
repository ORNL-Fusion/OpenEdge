if(PKG_VTK)
  # ######### BEGIN TARGET_SPARTA_PKG_VTK ##########

  if(NOT TARGET_SPARTA_PKG_VTK)
    message(FATAL_ERROR
      "PKG_VTK is ON here but TARGET_SPARTA_PKG_VTK is unset -- the "
      "PKG_VTK block in cmake/common/process/sparta_build_options.cmake "
      "(which sets it) must have been skipped or failed. That file now "
      "has an earlier, clearer check for this; if you're hitting this "
      "message it should have fired first -- check the full configure "
      "log above for a VTK-related FATAL_ERROR before this point.")
  endif()

  file(
    GLOB SPARTA_PKG_VTK_SRC_FILES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS *.cpp *.hpp *.h)

  include(${SPARTA_CMAKE_COMMON_DIR}/set/style_file_glob.cmake)

  add_library(${TARGET_SPARTA_PKG_VTK} STATIC ${SPARTA_PKG_VTK_SRC_FILES})
  set_target_properties(${TARGET_SPARTA_PKG_VTK} PROPERTIES LINKER_LANGUAGE CXX)

  # Get MPI header files
  target_link_libraries(${TARGET_SPARTA_PKG_VTK} LINK_PRIVATE
                        ${TARGET_SPARTA_BUILD_MPI})

  # Get Kokkos header files (only when a Kokkos build target exists)
  if(TARGET ${TARGET_SPARTA_BUILD_KOKKOS})
    target_link_libraries(${TARGET_SPARTA_PKG_VTK} LINK_PRIVATE
                          ${TARGET_SPARTA_BUILD_KOKKOS})
  endif()

  # Get VTK header files and libraries
  target_link_libraries(${TARGET_SPARTA_PKG_VTK} LINK_PUBLIC
                        ${TARGET_SPARTA_BUILD_VTK})

  # Make include public so that targets which link against this can find the
  # includes
  target_include_directories(${TARGET_SPARTA_PKG_VTK}
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  # Add include dependencies for building TARGET_SPARTA_PKG_VTK
  target_include_directories(${TARGET_SPARTA_PKG_VTK}
                             PRIVATE ${SPA_INCLUDE_DIRS})

  # VTK 7 exports plain library names without attached include dirs
  # (VTK 9 imported targets carry them); add them explicitly there
  if(VTK_VERSION VERSION_LESS "8.90")
    target_include_directories(${TARGET_SPARTA_PKG_VTK}
                               PRIVATE ${VTK_INCLUDE_DIRS})
  endif()

  # VTK 9 module system requires autoinit so the IO writer object factories
  # register at load time; without this the XML/legacy writers produce no
  # output. VTK 7 has no such command (and needs no autoinit for writers).
  if(COMMAND vtk_module_autoinit)
    vtk_module_autoinit(TARGETS ${TARGET_SPARTA_PKG_VTK}
                        MODULES ${TARGET_SPARTA_BUILD_VTK})
  endif()

  install(TARGETS ${TARGET_SPARTA_PKG_VTK}
          DESTINATION lib/sparta/${SPARTA_MACHINE}/pkgs)
  # ######### END   TARGET_SPARTA_PKG_VTK ##########
endif()

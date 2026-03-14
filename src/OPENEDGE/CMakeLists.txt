if(PKG_OPENEDGE)
  # ######### BEGIN TARGET_SPARTA_PKG_OPENEDGE ##########

  # Copy OpenEdge Kokkos files to src/KOKKOS/ so they are picked up
  # by the KOKKOS package glob when both PKG_OPENEDGE and PKG_KOKKOS are ON.
  if(PKG_KOKKOS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/KOKKOS")
    file(GLOB _OE_KOKKOS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/KOKKOS/*")
    foreach(_f ${_OE_KOKKOS_FILES})
      get_filename_component(_fname ${_f} NAME)
      configure_file(${_f} "${CMAKE_CURRENT_SOURCE_DIR}/../KOKKOS/${_fname}" COPYONLY)
    endforeach()
    message(STATUS "OpenEdge: copied Kokkos files to src/KOKKOS/")
  endif()

  file(
    GLOB SPARTA_PKG_OPENEDGE_SRC_FILES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS *.cpp *.hpp *.h)

  # Exclude override files that replace base SPARTA files.
  # These are installed into src/ via Install.sh (make yes-openedge)
  # and compiled as part of the main SPARTA library, not this package.
  set(SPARTA_PKG_OPENEDGE_OVERRIDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/update.h
    ${CMAKE_CURRENT_SOURCE_DIR}/input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/input.h
    ${CMAKE_CURRENT_SOURCE_DIR}/particle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/variable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/variable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_grid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_particle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_particle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sparta.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparta.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_emit_face_file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_emit_face_file.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_emit_surf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_emit_surf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_field_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_field_grid.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_field_particle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fix_field_particle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/surf_collide_diffuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/surf_collide_diffuse.h)
  list(REMOVE_ITEM SPARTA_PKG_OPENEDGE_SRC_FILES
       ${SPARTA_PKG_OPENEDGE_OVERRIDE_FILES})

  include(${SPARTA_CMAKE_COMMON_DIR}/set/style_file_glob.cmake)

  add_library(${TARGET_SPARTA_PKG_OPENEDGE} STATIC
              ${SPARTA_PKG_OPENEDGE_SRC_FILES})
  set_target_properties(${TARGET_SPARTA_PKG_OPENEDGE}
                        PROPERTIES LINKER_LANGUAGE CXX)

  # Get MPI header files
  target_link_libraries(${TARGET_SPARTA_PKG_OPENEDGE} LINK_PRIVATE
                        ${TARGET_SPARTA_BUILD_MPI})

  # Use PRIVATE include to avoid header collisions with override files in src/.
  # New OpenEdge headers are available in src/ (committed to the repo or
  # installed via make yes-openedge).
  target_include_directories(${TARGET_SPARTA_PKG_OPENEDGE}
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  # Add include dependencies for building TARGET_SPARTA_PKG_OPENEDGE
  target_include_directories(${TARGET_SPARTA_PKG_OPENEDGE}
                             PRIVATE ${SPA_INCLUDE_DIRS})

  install(TARGETS ${TARGET_SPARTA_PKG_OPENEDGE}
          DESTINATION lib/sparta/${SPARTA_MACHINE}/pkgs)
  # ######### END   TARGET_SPARTA_PKG_OPENEDGE ##########
endif()

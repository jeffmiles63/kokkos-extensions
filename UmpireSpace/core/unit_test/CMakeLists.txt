
FOREACH(Tag Threads;Serial;OpenMP;Cuda;HPX)
  # Because there is always an exception to the rule
  STRING(TOLOWER ${Tag} dir)
  SET(SRC_NAME ${Tag}_SOURCES)
  IF (Tag STREQUAL "Serial" OR Tag STREQUAL "Cuda")
     SET(SRC_NAME ${Tag}_SOURCES2)
  ENDIF()
  LIST(APPEND ${SRC_NAME}
     ${CMAKE_CURRENT_LIST_DIR}/${dir}/Test${Tag}_UmpireMemorySpace.cpp)
ENDFOREACH()


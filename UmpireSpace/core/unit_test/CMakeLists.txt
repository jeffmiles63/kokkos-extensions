
FOREACH(Tag Threads;Serial;OpenMP;Cuda;HPX)
  # Because there is always an exception to the rule
  STRING(TOLOWER ${Tag} dir)
  LIST(APPEND ${Tag}_SOURCES
     ${CMAKE_CURRENT_LIST_DIR}/${dir}/Test${Tag}_UmpireMemorySpace.cpp)
ENDFOREACH()


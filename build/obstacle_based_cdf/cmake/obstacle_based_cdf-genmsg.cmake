# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "obstacle_based_cdf: 1 messages, 0 services")

set(MSG_I_FLAGS "-Iobstacle_based_cdf:/home/maslab1/L-CDF/src/obstacle_based_cdf/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(obstacle_based_cdf_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_custom_target(_obstacle_based_cdf_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "obstacle_based_cdf" "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(obstacle_based_cdf
  "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/obstacle_based_cdf
)

### Generating Services

### Generating Module File
_generate_module_cpp(obstacle_based_cdf
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/obstacle_based_cdf
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(obstacle_based_cdf_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(obstacle_based_cdf_generate_messages obstacle_based_cdf_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_dependencies(obstacle_based_cdf_generate_messages_cpp _obstacle_based_cdf_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(obstacle_based_cdf_gencpp)
add_dependencies(obstacle_based_cdf_gencpp obstacle_based_cdf_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS obstacle_based_cdf_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(obstacle_based_cdf
  "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/obstacle_based_cdf
)

### Generating Services

### Generating Module File
_generate_module_eus(obstacle_based_cdf
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/obstacle_based_cdf
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(obstacle_based_cdf_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(obstacle_based_cdf_generate_messages obstacle_based_cdf_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_dependencies(obstacle_based_cdf_generate_messages_eus _obstacle_based_cdf_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(obstacle_based_cdf_geneus)
add_dependencies(obstacle_based_cdf_geneus obstacle_based_cdf_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS obstacle_based_cdf_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(obstacle_based_cdf
  "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/obstacle_based_cdf
)

### Generating Services

### Generating Module File
_generate_module_lisp(obstacle_based_cdf
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/obstacle_based_cdf
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(obstacle_based_cdf_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(obstacle_based_cdf_generate_messages obstacle_based_cdf_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_dependencies(obstacle_based_cdf_generate_messages_lisp _obstacle_based_cdf_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(obstacle_based_cdf_genlisp)
add_dependencies(obstacle_based_cdf_genlisp obstacle_based_cdf_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS obstacle_based_cdf_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(obstacle_based_cdf
  "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/obstacle_based_cdf
)

### Generating Services

### Generating Module File
_generate_module_nodejs(obstacle_based_cdf
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/obstacle_based_cdf
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(obstacle_based_cdf_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(obstacle_based_cdf_generate_messages obstacle_based_cdf_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_dependencies(obstacle_based_cdf_generate_messages_nodejs _obstacle_based_cdf_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(obstacle_based_cdf_gennodejs)
add_dependencies(obstacle_based_cdf_gennodejs obstacle_based_cdf_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS obstacle_based_cdf_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(obstacle_based_cdf
  "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/obstacle_based_cdf
)

### Generating Services

### Generating Module File
_generate_module_py(obstacle_based_cdf
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/obstacle_based_cdf
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(obstacle_based_cdf_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(obstacle_based_cdf_generate_messages obstacle_based_cdf_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/maslab1/L-CDF/src/obstacle_based_cdf/msg/Array.msg" NAME_WE)
add_dependencies(obstacle_based_cdf_generate_messages_py _obstacle_based_cdf_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(obstacle_based_cdf_genpy)
add_dependencies(obstacle_based_cdf_genpy obstacle_based_cdf_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS obstacle_based_cdf_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/obstacle_based_cdf)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/obstacle_based_cdf
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(obstacle_based_cdf_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/obstacle_based_cdf)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/obstacle_based_cdf
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(obstacle_based_cdf_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/obstacle_based_cdf)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/obstacle_based_cdf
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(obstacle_based_cdf_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/obstacle_based_cdf)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/obstacle_based_cdf
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(obstacle_based_cdf_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/obstacle_based_cdf)
  install(CODE "execute_process(COMMAND \"/home/maslab1/anaconda3/envs/gxf/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/obstacle_based_cdf\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/obstacle_based_cdf
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(obstacle_based_cdf_generate_messages_py std_msgs_generate_messages_py)
endif()

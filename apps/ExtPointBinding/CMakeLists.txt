project(ExtPointBinding)

set(Demo_ExtPointBinding_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/external_point_binding_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/point_conversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/point_extlib1.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/point_extlib2.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pointadapter_extlib1.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pointadapter_extlib2.hpp
)

add_executable(${PROJECT_NAME} ${Demo_ExtPointBinding_SRC})
target_link_libraries(${PROJECT_NAME} gr::utils gr::accel gr::algo gr::io)
add_dependencies(${PROJECT_NAME} opengr)
install( TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin  )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

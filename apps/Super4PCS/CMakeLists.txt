project(Super4PCS)

set(Demo_Super4PCS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/super4pcs_test.cc
)

add_executable(${PROJECT_NAME} ${Demo_Super4PCS_SRC})
target_link_libraries(${PROJECT_NAME} gr::utils gr::accel gr::algo gr::io)
add_dependencies(${PROJECT_NAME} opengr)
install( TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin  )
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

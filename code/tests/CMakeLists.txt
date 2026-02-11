# EulerIntegration tests
add_executable(test_euler_integration
    integration_schemes/test_euler_integration.cpp
)

target_include_directories(test_euler_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(test_euler_integration PRIVATE
    ${PROJECT_NAME}
    GTest::gtest
    GTest::gtest_main
)

# Register tests with CTest
gtest_discover_tests(test_euler_integration
    PROPERTIES LABELS "integration_schemes"
)

# InvDynamics tests
add_executable(test_inverse_dynamics
    constraints/test_inverse_dynamics.cpp
)

target_include_directories(test_inverse_dynamics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(test_inverse_dynamics PRIVATE
    ${PROJECT_NAME}
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_inverse_dynamics
    PROPERTIES LABELS "constraints"
)

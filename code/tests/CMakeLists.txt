function(add_trajopt_test target source label)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    target_link_libraries(${target} PRIVATE
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )
    gtest_discover_tests(${target} PROPERTIES LABELS ${label})
endfunction()

file(GLOB_RECURSE TEST_SOURCES "**.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    # Derive target name from file path relative to this directory
    file(RELATIVE_PATH rel ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SOURCE})
    string(REPLACE "/" "_" target ${rel})
    string(REPLACE ".cpp" "" target ${target})

    # Use the subdirectory as the label
    get_filename_component(subdir ${rel} DIRECTORY)
    if(subdir STREQUAL "")
        set(label "misc")
    else()
        set(label ${subdir})
    endif()

    add_trajopt_test(${target} ${TEST_SOURCE} ${label})
endforeach()

cmake_minimum_required(VERSION 3.16)

project(gui VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
        main.cpp
        mainwindow/mainwindow.cpp
        mainwindow/mainwindow.h
        mainwindow/mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Core)

    qt_add_executable(gui
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        version.h
        tabs/prepare/preparetab.h tabs/prepare/preparetab.cpp tabs/prepare/preparetab.ui
        tabs/preview/previewtab.h tabs/preview/previewtab.cpp tabs/preview/previewtab.ui
        tabs/monitor/monitortab.h tabs/monitor/monitortab.cpp tabs/monitor/monitortab.ui
        core/appconfig.h core/appconfig.cpp
        core/model3d.h core/model3d.cpp
        widgets/setupWizard/setupwizard.h widgets/setupWizard/setupwizard.cpp
        widgets/setupWizard/pages/welcomepage/welcomepage.h widgets/setupWizard/pages/welcomepage/welcomepage.cpp widgets/setupWizard/pages/welcomepage/welcomepage.ui
        widgets/setupWizard/pages/profilepage/profilepage.h widgets/setupWizard/pages/profilepage/profilepage.cpp widgets/setupWizard/pages/profilepage/profilepage.ui
        resource.qrc
        tabs/prepare/settingsMenuWidget/settingsmenuwidget.h tabs/prepare/settingsMenuWidget/settingsmenuwidget.cpp tabs/prepare/settingsMenuWidget/settingsmenuwidget.ui
        core/profiles/printerprofile.h core/profiles/printerprofile.cpp
        core/profiles/materialprofile.h core/profiles/materialprofile.cpp
        core/profiles/processprofile.h core/profiles/processprofile.cpp
        core/profilemanager.h core/profilemanager.cpp
    )
    target_include_directories(gui PRIVATE
        mainwindow
        core
        core/profiles
        tabs/prepare
        tabs/prepare/settingsMenuWidget
        tabs/preview
        tabs/monitor
        widgets/setupWizard
        widgets/setupWizard/pages/welcomepage
        widgets/setupWizard/pages/profilepage

    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET gui APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(gui SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(gui
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(gui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(gui PRIVATE Qt6::Core)
target_link_libraries(gui PRIVATE Qt6::Core)
target_link_libraries(gui PRIVATE Qt6::Core)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.gui)
endif()
set_target_properties(gui PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS gui
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(gui)
endif()

import QtQuick 2.15
import QtQuick3D
View3D {
    id: view
    anchors.fill: parent

    environment: SceneEnvironment {
        clearColor: "#1e1e1e"
        backgroundMode: SceneEnvironment.Color
    }

    PerspectiveCamera {
        id: camera
        position: Qt.vector3d(0, 0, 300)
    }

    DirectionalLight {
        eulerRotation.x: -45
        eulerRotation.y: 45
        brightness: 1.2
    }

    Model {
        id: model
        source: ""             // set from C++
        scale: Qt.vector3d(0.1,0.1,0.1)
        position: Qt.vector3d(0,0,0)
        materials: DefaultMaterial {
            diffuseColor: "lightgray"
        }
    }
}

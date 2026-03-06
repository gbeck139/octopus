import QtQuick 2.15
import QtQuick.Controls 2.15
import Qt3D.Core 2.15
import Qt3D.Render 2.15
import Qt3D.Extras 2.15

Rectangle {
    id: page
    width: 800
    height: 600
    color: "lightgray"

    // Title
    Text {
        id: titleText
        text: "3D Modeller"
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 20
        font.pointSize: 24
        font.bold: true
        color: "black"
    }

    // 3D viewer area
    Item {
        id: viewerArea
        anchors.top: titleText.bottom
        anchors.topMargin: 10
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom

        // Qt3D Scene root
        Entity {
            id: sceneRoot

            // Camera
            Camera {
                id: camera
                position: Qt.vector3d(0, 100, 600)
                viewCenter: Qt.vector3d(0, 50, 0)
                upVector: Qt.vector3d(0, 1, 0)
                fieldOfView: 45
                nearPlane: 0.1
                farPlane: 2000.0
            }

            // Orbit camera controller
            OrbitCameraController {
                camera: camera
                linearSpeed: 50
                lookSpeed: 180
            }

            // Directional light
            DirectionalLight {
                worldDirection: Qt.vector3d(-1, -1, -2)
                color: "white"
                intensity: 1.0
            }

            // STL mesh
            Mesh {
                id: stlMesh
                source: "C:/Users/canca/Downloads/3DBenchy.stl" // <-- change path
            }

            PhongMaterial {
                id: stlMaterial
                diffuse: "lightsteelblue"
                specular: "white"
                shininess: 50
            }

            Transform {
                id: stlTransform
                scale3D: Qt.vector3d(1, 1, 1)
                rotation: Qt.quaternionFromEulerAngles(0, 0, 0)
                translation: Qt.vector3d(0, 0, 0)
            }

            Entity {
                id: stlEntity
                components: [
                    stlMesh,
                    stlMaterial,
                    stlTransform
                ]
            }

            // Optional ground plane
            PlaneMesh {
                id: groundPlane
                width: 1000
                height: 1000
            }

            Transform {
                id: groundTransform
                translation: Qt.vector3d(0, -100, 0)
                rotation: Qt.quaternionFromEulerAngles(-90, 0, 0)
            }

            PhongMaterial {
                id: groundMaterial
                diffuse: "lightgray"
            }

            Entity {
                id: groundEntity
                components: [
                    groundPlane,
                    groundMaterial,
                    groundTransform
                ]
            }
        }
    }
}

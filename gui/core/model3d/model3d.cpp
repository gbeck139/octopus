#include "model3d.h"

#include <QDebug>
#include <QDir>
#include <QMatrix4x4>

Model3D::Model3D(QObject *parent)
    : QObject{parent},
    boxMin(0,0,0),
    boxMax(0,0,0)
{}

bool Model3D::loadModel(const QString &filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "ERROR: Failed to open STL file:" << filePath;
        return false;
    }

    clearModel();

    // Make a temp copy of the STL with the same base name
    QFileInfo info(filePath);
    QString tempPath = QDir::temp().filePath(info.completeBaseName() + ".stl");

    if (QFile::exists(tempPath))
        QFile::remove(tempPath);

    if (!QFile::copy(filePath, tempPath)) {
        qDebug() << "[Model3D] Failed to copy STL to temp:" << tempPath;
        return false;
    }

    qDebug() << "[Model3D] Temp STL created:" << tempPath;

    // Now treat tempPath as the "source"
    sourceFilePath = tempPath;

    // Detect ASCII vs Binary from the original file, then load
    QByteArray start = file.peek(256);
    bool isAscii = start.startsWith("solid") && start.contains("facet");
    bool success = isAscii ? loadAsciiSTL(file) : loadBinarySTL(file);

    file.close();
    if (!success || meshTriangles.isEmpty()) {
        qDebug() << "ERROR: No triangles loaded from STL";
        return false;
    }

    computeBoundingBox();
    emit modelChanged();

    return true;
}

void Model3D::clearModel()
{
    meshTriangles.clear();
    sourceFilePath.clear();
    boxMin = QVector3D(0,0,0);
    boxMax = QVector3D(0,0,0);

    emit modelChanged();
}

bool Model3D::isLoaded() const
{
    return !meshTriangles.isEmpty();
}

QString Model3D::getSourceFilePath() const
{
    return sourceFilePath;
}

const QVector<Triangle> &Model3D::getTriangles() const
{
    return meshTriangles;
}

QVector3D Model3D::boundingBoxMin() const
{
    return boxMin;
}

QVector3D Model3D::boundingBoxMax() const
{
    return boxMax;
}

void Model3D::computeBoundingBox()
{
    if (meshTriangles.isEmpty()) {
        qDebug() << "ERROR: mesh triangles are empty";
        return;
    }

    float minX = meshTriangles[0].v1.x();
    float minY = meshTriangles[0].v1.y();
    float minZ = meshTriangles[0].v1.z();
    float maxX = minX;
    float maxY = minY;
    float maxZ = minZ;

    for (const auto& tri : meshTriangles) {
        for (const auto& v : { tri.v1, tri.v2, tri.v3 }) {
            minX = qMin(minX, v.x());
            minY = qMin(minY, v.y());
            minZ = qMin(minZ, v.z());
            maxX = qMax(maxX, v.x());
            maxY = qMax(maxY, v.y());
            maxZ = qMax(maxZ, v.z());
        }
    }

    boxMin = QVector3D(minX, minY, minZ);
    boxMax = QVector3D(maxX, maxY, maxZ);
}

bool Model3D::loadAsciiSTL(QFile &file)
{
    QTextStream in(&file);
    QVector<QVector3D> tempVerts;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();

        if (line.startsWith("vertex")) {
            QStringList parts = line.split(" ", Qt::SkipEmptyParts);
            if (parts.size() == 4) {
                float x = parts[1].toFloat();
                float y = parts[2].toFloat();
                float z = parts[3].toFloat();
                tempVerts.append(QVector3D(x, y, z));

                if (tempVerts.size() == 3) {
                    meshTriangles.append({ tempVerts[0], tempVerts[1], tempVerts[2] });
                    tempVerts.clear();
                }
            }
        }
    }
    return true;
}

bool Model3D::loadBinarySTL(QFile &file)
{
    file.seek(80); // skip header
    quint32 triangleCount;

    if (file.read(reinterpret_cast<char*>(&triangleCount), sizeof(triangleCount)) != sizeof(triangleCount)) {
        return false;
    }

    for (quint32 i = 0; i < triangleCount; ++i) {
        float normal[3], verts[9];
        quint16 attr;

        if (file.read(reinterpret_cast<char*>(normal), sizeof(normal)) != sizeof(normal)) break;
        if (file.read(reinterpret_cast<char*>(verts), sizeof(verts)) != sizeof(verts)) break;
        if (file.read(reinterpret_cast<char*>(&attr), sizeof(attr)) != sizeof(attr)) break;

        meshTriangles.append({
            QVector3D(verts[0], verts[1], verts[2]),
            QVector3D(verts[3], verts[4], verts[5]),
            QVector3D(verts[6], verts[7], verts[8])
        });
    }
    return true;
}

void Model3D::rotateToFace(const QString &face)
{
    if (meshTriangles.isEmpty()) return;

    QMatrix4x4 mat;
    if (face == "Front") mat.setToIdentity();
    else if (face == "Back") mat.rotate(180, 0, 1, 0);
    else if (face == "Top") mat.rotate(-90, 1, 0, 0);
    else if (face == "Bottom") mat.rotate(90, 1, 0, 0);
    else if (face == "Left") mat.rotate(-90, 0, 1, 0);
    else if (face == "Right") mat.rotate(90, 0, 1, 0);

    for (auto &tri : meshTriangles) {
        tri.v1 = mat * tri.v1;
        tri.v2 = mat * tri.v2;
        tri.v3 = mat * tri.v3;
    }

    computeBoundingBox();
    emit modelChanged();
}

bool Model3D::saveTempSTL()
{
    if (meshTriangles.isEmpty() || sourceFilePath.isEmpty()) return false;

    QFile file(sourceFilePath);
    if (!file.open(QIODevice::WriteOnly)) return false;

    QTextStream out(&file);
    out << "solid model\n";
    for (const auto &tri : meshTriangles) {
        out << "  facet normal 0 0 0\n";
        out << "    outer loop\n";
        out << "      vertex " << tri.v1.x() << " " << tri.v1.y() << " " << tri.v1.z() << "\n";
        out << "      vertex " << tri.v2.x() << " " << tri.v2.y() << " " << tri.v2.z() << "\n";
        out << "      vertex " << tri.v3.x() << " " << tri.v3.y() << " " << tri.v3.z() << "\n";
        out << "    endloop\n";
        out << "  endfacet\n";
    }
    out << "endsolid model\n";
    file.close();
    return true;
}

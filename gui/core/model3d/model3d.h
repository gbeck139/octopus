#ifndef MODEL3D_H
#define MODEL3D_H

#include <QObject>
#include <QVector3D>
#include <QVector>
#include <QString>
#include <QFile>

///
/// \brief The Triangle class contains three verticies
///
struct Triangle {
    QVector3D v1;
    QVector3D v2;
    QVector3D v3;
};

///
/// \brief The Model3D class loads and stores information from
/// a 3D model from an STL file. Supports both ASCII and Binary STL.
///
class Model3D : public QObject
{
    Q_OBJECT
public:
    explicit Model3D(QObject *parent = nullptr);

    // Load and Clear
    bool loadModel(const QString& filePath);
    void clearModel();

    // Model info
    bool isLoaded() const;
    QString getSourceFilePath() const;
    const QVector<Triangle>& getTriangles() const;

    // Bounding box limits
    QVector3D boundingBoxMin() const;
    QVector3D boundingBoxMax() const;

    // Operations
    // void cleanModel //clean/fix geometry

signals:
    void modelChanged(); //when model's geometry changes


private:
    ///
    /// \brief computeBoundingBox finds the smallest rectangular box
    /// that completely contains the 3D model
    ///
    void computeBoundingBox();
    bool loadAsciiSTL(QFile &file);
    bool loadBinarySTL(QFile &file);

private:
    QVector<Triangle> meshTriangles;
    QString sourceFilePath;

    ///
    /// \brief boxMin is the corner point on the 3D model
    /// with the smallest x, y, z
    ///
    QVector3D boxMin;

    ///
    /// \brief boxMax is the corner point on the 3D model
    /// with the largest x, y, z
    ///
    QVector3D boxMax;
};

#endif // MODEL3D_H

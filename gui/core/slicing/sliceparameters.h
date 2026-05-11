#ifndef SLICEPARAMETERS_H
#define SLICEPARAMETERS_H

#include <QString>
#include <QStringList>
#include <QJsonObject>

class SliceParameters
{
public:

    enum class SlicingMode
    {
        Planar,
        RadialNonPlanar
    };

    // =========================
    // MODEL
    // =========================
    double rotX = 0.0;
    double rotY = 0.0;
    double rotZ = 0.0;

    QString modelName;

    // =========================
    // FILE PATHS
    // =========================
    QString stlPath;
    QString prusaSlicerPath;
    QString outputDirectory;

    // =========================
    // DEFORMATION
    // =========================
    double angleBase = 15.0;
    double angleFactor = 30.0;

    // =========================
    // PRINT
    // =========================
    double layerHeight = 0.2;

    int nozzleTemperature = 210;
    int bedTemperature = 60;

    int infillDensity = 20;
    int wallLoops = 2;
    bool supportsEnabled = false;

    // =========================
    // MODE
    // =========================
    SlicingMode slicingMode = SlicingMode::RadialNonPlanar;

    // =========================
    // CORE API
    // =========================
    bool validate(QStringList& errors) const;

    QJsonObject toJson() const;

    bool saveToFile(const QString& filePath) const;

    static SliceParameters fromJson(const QJsonObject& obj);
};

#endif // SLICEPARAMETERS_H

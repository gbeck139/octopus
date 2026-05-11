#ifndef SLICEPARAMETERS_H
#define SLICEPARAMETERS_H

#include <QString>
#include <QStringList>
#include <QJsonObject>
#include <QJsonDocument>

class SliceParameters
{
public:

    // =========================
    // ENUMS
    // =========================

    enum class SlicingMode
    {
        Planar,
        RadialNonPlanar
    };

    // =========================
    // MODEL SETTINGS
    // =========================

    double rotX = 0.0;
    double rotY = 0.0;
    double rotZ = 0.0;

    // =========================
    // DEFORMATION SETTINGS
    // =========================

    double angleBase = 15.0;
    double angleFactor = 30.0;

    // =========================
    // PRINT SETTINGS
    // =========================

    double layerHeight = 0.2;

    int nozzleTemperature = 210;
    int bedTemperature = 60;

    int infillDensity = 20;

    int wallLoops = 2;

    bool supportsEnabled = false;

    // =========================
    // FILE PATHS
    // =========================

    QString stlPath;
    QString outputDirectory;
    QString prusaSlicerPath;

    // =========================
    // PIPELINE SETTINGS
    // =========================

    SlicingMode slicingMode = SlicingMode::RadialNonPlanar;

    // =========================
    // VALIDATION
    // =========================

    bool validate(QStringList& errors) const;

    // =========================
    // JSON
    // =========================

    QJsonObject toJson() const;

    bool saveToFile(const QString& filePath) const;

    static SliceParameters fromJson(const QJsonObject& obj);

private:

    QString slicingModeToString() const;
};

#endif // SLICEPARAMETERS_H
